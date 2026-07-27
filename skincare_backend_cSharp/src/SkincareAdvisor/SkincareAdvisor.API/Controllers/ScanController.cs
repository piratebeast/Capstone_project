using System;
using System.Diagnostics; // <-- REQUIRED FOR HIGH-PRECISION STOPWATCH TIMING
using System.IO;
using System.Linq;
using System.Security.Claims;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.EntityFrameworkCore;
using SkincareAdvisor.Application.DTOs;
using SkincareAdvisor.Application.Interfaces;
using SkincareAdvisor.Domain.Entities;
using SkincareAdvisor.Infrastructure.Persistence;

namespace SkincareAdvisor.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]
    [EnableRateLimiting("api-policy")]
    public class ScanController : ControllerBase
    {
        private readonly IScanService _scanService;
        private readonly ApplicationDbContext _context;
        private readonly IScanCritiqueService _scanCritiqueService;

        public ScanController(IScanService scanService, ApplicationDbContext context, IScanCritiqueService scanCritiqueService)
        {
            _scanService = scanService;
            _context = context;
            _scanCritiqueService = scanCritiqueService; 
        }

        [HttpPost("analyze")]
        public async Task<IActionResult> AnalyzeFace([FromForm] ScanRequest request)
        {
            if (request.image == null || request.image.Length == 0)
                return BadRequest("No image was uploaded.");

            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userId))
                return Unauthorized("User is not authenticated.");

            // Initialize background telemetry tracking logs framework
            var stopwatch = new Stopwatch();
            var performanceLog = new SystemPerformanceLog
            {
                UserId = userId,
                Timestamp = DateTime.UtcNow,
                Endpoint = "api/Scan/analyze"
            };

            try
            {
                // 1. Calculate Age
                var user = await _context.Users.FindAsync(userId);
                int calculatedAge = 25;
                if (user != null && user.DateOfBirth.HasValue)
                {
                    var today = DateTime.Today;
                    var dob = user.DateOfBirth.Value;
                    calculatedAge = today.Year - dob.Year;
                    if (dob.Date > today.AddYears(-calculatedAge)) calculatedAge--;
                    if (calculatedAge <= 0) { calculatedAge = 25; }
                }

                // 2. Permanently Save the Original Image File to Disk
                var uniqueFileName = $"{Guid.NewGuid()}{Path.GetExtension(request.image.FileName)}";
                var targetStorageFolder = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads", "scans");
                if (!Directory.Exists(targetStorageFolder))
                {
                    Directory.CreateDirectory(targetStorageFolder);
                }

                var fullPhysicalWritePath = Path.Combine(targetStorageFolder, uniqueFileName);
                using (var fileStream = new FileStream(fullPhysicalWritePath, FileMode.Create))
                {
                    await request.image.CopyToAsync(fileStream);
                }

                var savedWebImageUrl = $"/uploads/scans/{uniqueFileName}";

                // 3. START TIMING: Call the Python FastAPI Server 
                stopwatch.Start();
                AiScanResponse aiResult = await _scanService.AnalyzeImageAsync(request.image, calculatedAge);
                stopwatch.Stop();

                // 4. Map the Extended DTO into our Database Entity
                var scanHistory = new ScanHistory
                {
                    UserId = userId,
                    ImageUrl = savedWebImageUrl,
                    Acne = aiResult.Diagnostics.Acne,
                    DarkSpots = aiResult.Diagnostics.DarkSpots,
                    Wrinkles = aiResult.Diagnostics.Wrinkles,
                    Redness = aiResult.Diagnostics.Redness,
                    DarkCircles = aiResult.Diagnostics.DarkCircles,
                    Gender = aiResult.Diagnostics.Gender,
                    RoutineClass = aiResult.RoutineClass,
                    Confidence = aiResult.Confidence,

                    AcneHeatmap = aiResult.Heatmaps.Acne,
                    DarkSpotsHeatmap = aiResult.Heatmaps.DarkSpots,
                    WrinklesHeatmap = aiResult.Heatmaps.Wrinkles,
                    RednessHeatmap = aiResult.Heatmaps.Redness,
                    DarkCirclesHeatmap = aiResult.Heatmaps.DarkCircles,

                    DailyAm = aiResult.RegimenSchedule.DailyAm.Select(a => new RoutineStepEntity
                    {
                        Step = a.Step,
                        Product = a.Product,
                        Purpose = a.Purpose
                    }).ToList(),

                    DailyPm = aiResult.RegimenSchedule.DailyPm.Select(p => new RoutineStepEntity
                    {
                        Step = p.Step,
                        Product = p.Product,
                        Purpose = p.Purpose
                    }).ToList(),

                    WeeklyTreatments = aiResult.RegimenSchedule.WeeklyTreatments.Select(w => new WeeklyTreatmentEntity
                    {
                        Product = w.Product,
                        Frequency = w.Frequency,
                        Instructions = w.Instructions
                    }).ToList()
                };

                // Append successful analytics states
                performanceLog.InferenceLatencyMs = (int)stopwatch.ElapsedMilliseconds;
                performanceLog.IsSuccess = true;

                // 5. Save both histories and telemetry blocks simultaneously
                _context.ScanHistories.Add(scanHistory);
                _context.SystemPerformanceLogs.Add(performanceLog);
                await _context.SaveChangesAsync();

                return Ok(new
                {
                    Message = "Scan complete and saved successfully!",
                    ScanId = scanHistory.Id,
                    Data = new
                    {
                        Diagnostics = aiResult.Diagnostics,
                        Confidence = aiResult.Confidence,
                        ImageUrl = savedWebImageUrl
                    }
                });
            }
            // Catch MediaPipe Specific Fail-Fast Rejections
            catch (ArgumentException ex)
            {
                stopwatch.Stop();

                performanceLog.InferenceLatencyMs = (int)stopwatch.ElapsedMilliseconds;
                performanceLog.IsSuccess = false;
                performanceLog.FailureReason = ex.Message.Contains("Multiple")
                    ? "MULTIPLE_FACES_DETECTED"
                    : "NO_FACE_DETECTED";

                _context.SystemPerformanceLogs.Add(performanceLog);
                await _context.SaveChangesAsync(); // Saves the metric to feed your 4.2% dashboard graph

                return BadRequest(new { Error = "Invalid Image", Message = ex.Message });
            }
            // Catch Generic System Failures (Python Offline, Network dropped, etc.)
            catch (Exception ex)
            {
                stopwatch.Stop();

                performanceLog.InferenceLatencyMs = (int)stopwatch.ElapsedMilliseconds;
                performanceLog.IsSuccess = false;
                performanceLog.FailureReason = "SERVER_ERROR";

                _context.SystemPerformanceLogs.Add(performanceLog);
                await _context.SaveChangesAsync();

                return StatusCode(500, $"An error occurred during analysis: {ex.Message}");
            }
        }

        [HttpGet("routine/{scanId}")]
        public async Task<IActionResult> GetSpecificRoutine(Guid scanId)
        {
            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            var isAdmin = User.IsInRole("Admin");

            var scan = await _context.ScanHistories
                .FirstOrDefaultAsync(s => s.Id == scanId && (s.UserId == userId || isAdmin));

            if (scan == null)
            {
                return NotFound("That specific routine could not be found.");
            }

            return Ok(new
            {
                ScanDate = scan.ScanDate,
                RoutineClass = scan.RoutineClass,
                Confidence = scan.Confidence,
                RegimenSchedule = new
                {
                    DailyAm = scan.DailyAm,
                    DailyPm = scan.DailyPm,
                    WeeklyTreatments = scan.WeeklyTreatments
                }
            });
        }
    }
}