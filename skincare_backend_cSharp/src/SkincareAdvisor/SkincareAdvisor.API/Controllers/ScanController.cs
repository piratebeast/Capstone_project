using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
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
    [Authorize] // Ensure that only authenticated users can access this controller
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
            // 1. Validation
            if (request.image == null || request.image.Length == 0)
                return BadRequest("No image was uploaded.");

            try
            {
                // 2. Call the Python FastAPI Server (The Brains)
                AiScanResponse aiResult = await _scanService.AnalyzeImageAsync(request.image);

                // 3. Extract the logged-in User's ID directly from their JWT Token
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier) ?? "UnknownUser";

                // 4. Map the Python DTO into our Database Entity
                var scanHistory = new ScanHistory
                {
                    UserId = userId,
                    Acne = aiResult.Diagnostics.Acne,
                    DarkSpots = aiResult.Diagnostics.DarkSpots,
                    Wrinkles = aiResult.Diagnostics.Wrinkles,
                    Redness = aiResult.Diagnostics.Redness,
                    DarkCircles = aiResult.Diagnostics.DarkCircles,
                    Gender = aiResult.Diagnostics.Gender,
                    RoutineClass = aiResult.RoutineClass,
                    Confidence = aiResult.Confidence,

                    // Map the lists
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

                // 5. Save to SQL Server
                _context.ScanHistories.Add(scanHistory);
                await _context.SaveChangesAsync();

                // 6. Return the full custom routine to the Flutter App!
                return Ok(new
                {
                    Message = "Scan complete and saved successfully!",
                    ScanId = scanHistory.Id, // <-- CRITICAL: The frontend needs this ID
                    Data = new
                    {
                        Diagnostics = aiResult.Diagnostics,
                        Confidence = aiResult.Confidence
                    }
                });
            }
            catch (Exception ex)
            {
                // If Python is offline or crashes, catch the error gracefully
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
