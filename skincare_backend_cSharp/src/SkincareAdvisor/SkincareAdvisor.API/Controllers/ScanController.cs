using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.EntityFrameworkCore;
using SkincareAdvisor.Application.DTOs;
using SkincareAdvisor.Application.Interfaces;
using SkincareAdvisor.Domain.Entities;
using SkincareAdvisor.Infrastructure; // Assuming your ApplicationDbContext is here
using SkincareAdvisor.Infrastructure.Persistence;

namespace SkincareAdvisor.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]// Ensure that only authenticated users can access this controller
    [EnableRateLimiting("api-policy")] // Apply the rate limiting policy defined in Program.cs

    public class ScanController : ControllerBase
    {
        private readonly IScanService _scanService;
        private readonly ApplicationDbContext _context;

        // Inject both the ScanService and the Database Context
        public ScanController(IScanService scanService, ApplicationDbContext context)
        {
            _scanService = scanService;
            _context = context;
        }

        [HttpPost("analyze")]
        public async Task<IActionResult> AnalyzeFace([FromForm] ScanRequest request)
        {
            if (request.image == null || request.image.Length == 0)
                return BadRequest("No image was uploaded.");

            try
            {
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                if (string.IsNullOrEmpty(userId))
                    return Unauthorized("User is not authenticated.");

                // 1. Calculate Age (Keep your precise age deduction logic block)
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

                // ===================================================================
                // 2. NEW LOGIC: PERMANENTLY SAVE THE ORIGINAL IMAGE FILE TO DISK
                // ===================================================================
                var uniqueFileName = $"{Guid.NewGuid()}{Path.GetExtension(request.image.FileName)}";

                // Target physical path pointing into the wwwroot folder assets deployment map
                var targetStorageFolder = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads", "scans");
                if (!Directory.Exists(targetStorageFolder))
                {
                    Directory.CreateDirectory(targetStorageFolder);
                }

                var fullPhysicalWritePath = Path.Combine(targetStorageFolder, uniqueFileName);

                // Stream copy the bytes straight out of memory to write to disk
                using (var fileStream = new FileStream(fullPhysicalWritePath, FileMode.Create))
                {
                    await request.image.CopyToAsync(fileStream);
                }

                // Web address relative pointer url path string stored into the relational row
                var savedWebImageUrl = $"/uploads/scans/{uniqueFileName}";
                // ===================================================================

                // 3. Call the Python FastAPI Server 
                AiScanResponse aiResult = await _scanService.AnalyzeImageAsync(request.image, calculatedAge);

                // 4. Map the Extended DTO into our Database Entity
                var scanHistory = new ScanHistory
                {
                    UserId = userId,
                    ImageUrl = savedWebImageUrl, // <-- Save file reference link string
                    Acne = aiResult.Diagnostics.Acne,
                    DarkSpots = aiResult.Diagnostics.DarkSpots,
                    Wrinkles = aiResult.Diagnostics.Wrinkles,
                    Redness = aiResult.Diagnostics.Redness,
                    DarkCircles = aiResult.Diagnostics.DarkCircles,
                    Gender = aiResult.Diagnostics.Gender,
                    RoutineClass = aiResult.RoutineClass,
                    Confidence = aiResult.Confidence,

                    // NEW: Maps the in-memory matrix list channels through our Value Converter backings
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

                // 5. Save Transaction directly inside SQL Server
                _context.ScanHistories.Add(scanHistory);
                await _context.SaveChangesAsync();

                return Ok(new
                {
                    Message = "Scan complete and saved successfully!",
                    ScanId = scanHistory.Id,
                    Data = new
                    {
                        Diagnostics = aiResult.Diagnostics,
                        Confidence = aiResult.Confidence,
                        ImageUrl = savedWebImageUrl // Ships the reference URL directly back to your client apps
                    }
                });
            }
            catch (ArgumentException ex)
            {
                return BadRequest(new { Error = "Invalid Image", Message = ex.Message });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"An error occurred during analysis: {ex.Message}");
            }
        }

        [HttpGet("routine/{scanId}")] // URL will be: api/Scan/routine/00d73612...
        public async Task<IActionResult> GetSpecificRoutine(Guid scanId)
        {
            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);

            // Find the EXACT scan by ID and ensure it belongs to this user
            var scan = await _context.ScanHistories
                .FirstOrDefaultAsync(s => s.Id == scanId && s.UserId == userId);

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
