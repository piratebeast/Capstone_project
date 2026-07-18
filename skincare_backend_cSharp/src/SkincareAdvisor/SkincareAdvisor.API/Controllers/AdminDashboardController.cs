using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SkincareAdvisor.Infrastructure.Persistence;
using SkincareAdvisor.Application.Interfaces;
using SkincareAdvisor.Domain.Entities;

namespace SkincareAdvisor.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize(Roles = "Admin")] // Lock this entire controller down to identities possessing the verified Admin token claim
    public class AdminDashboardController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly IScanCritiqueService _critiqueService;

        public AdminDashboardController(ApplicationDbContext context, IScanCritiqueService critiqueService)
        {
            _context = context;
            _critiqueService = critiqueService;
        }

        /// <summary>
        /// Fetches an expansive data transaction profile for your Angular Live Diagnostic Feed dashboard.
        /// URL Endpoint: GET api/AdminDashboard/scans/{scanId}
        /// </summary>
        [HttpGet("scans/{scanId}")]
        public async Task<IActionResult> GetScanDetailForAdmin(Guid scanId)
        {
            var scan = await _context.ScanHistories
                .FirstOrDefaultAsync(s => s.Id == scanId);

            if (scan == null)
            {
                return NotFound(new { Message = "The requested diagnostic record does not exist." });
            }

            // NEW: pull the latest critique for this scan
            var latestCritique = await _context.ScanCritiques
                .Where(c => c.ScanHistoryId == scanId)
                .OrderByDescending(c => c.GeneratedAt)
                .FirstOrDefaultAsync();

            return Ok(new
            {
                ScanId = scan.Id,
                UserId = scan.UserId,
                ScanDate = scan.ScanDate,
                RoutineClass = scan.RoutineClass,
                Confidence = scan.Confidence,
                OriginalImageUrl = scan.ImageUrl,
                Diagnostics = new
                {
                    Acne = scan.Acne,
                    DarkSpots = scan.DarkSpots,
                    Wrinkles = scan.Wrinkles,
                    Redness = scan.Redness,
                    DarkCircles = scan.DarkCircles,
                    Gender = scan.Gender
                },
                Heatmaps = new
                {
                    Acne = scan.AcneHeatmap,
                    DarkSpots = scan.DarkSpotsHeatmap,
                    Wrinkles = scan.WrinklesHeatmap,
                    Redness = scan.RednessHeatmap,
                    DarkCircles = scan.DarkCirclesHeatmap
                },
                // NEW: expose critique so the frontend can hydrate on load/refresh
                AiCritique = latestCritique?.CritiqueText,
                AiCritiqueSucceeded = latestCritique?.Succeeded,
                AiCritiqueGeneratedAt = latestCritique?.GeneratedAt
            });
        }

        /// <summary>
        /// Optional Utility: Fetches a global summary list of all scans for management logs overview.
        /// URL Endpoint: GET api/AdminDashboard/scans/summary
        /// </summary>
        [HttpGet("scans/summary")]
        public async Task<IActionResult> GetAllScansSummary()
        {
            var summaryList = await _context.ScanHistories
                .OrderByDescending(s => s.ScanDate)
                .Select(s => new
                {
                    ScanId = s.Id,
                    UserId = s.UserId,
                    ScanDate = s.ScanDate,
                    RoutineClass = s.RoutineClass,
                    Confidence = s.Confidence,
                    PrimaryConditionSeverity = s.Acne,
                    // NEW: latest critique text + success flag, computed inline (EF translates this to SQL, no N+1)
                    AiCritique = _context.ScanCritiques
                        .Where(c => c.ScanHistoryId == s.Id)
                        .OrderByDescending(c => c.GeneratedAt)
                        .Select(c => c.CritiqueText)
                        .FirstOrDefault(),
                    AiCritiqueSucceeded = _context.ScanCritiques
                        .Where(c => c.ScanHistoryId == s.Id)
                        .OrderByDescending(c => c.GeneratedAt)
                        .Select(c => (bool?)c.Succeeded)
                        .FirstOrDefault()
                })
                .ToListAsync();

            return Ok(summaryList);
        }

        /// <summary>
        /// Generates (or regenerates) an AI critique for a scan and stores it.
        /// URL Endpoint: POST api/AdminDashboard/scans/{scanId}/critique
        /// </summary>
        [HttpPost("scans/{scanId}/critique")]
        public async Task<IActionResult> GenerateCritique(Guid scanId)
        {
            var scan = await _context.ScanHistories
                .FirstOrDefaultAsync(s => s.Id == scanId);

            if (scan == null)
                return NotFound(new { Message = "The requested diagnostic record does not exist." });

            var critique = await _critiqueService.GenerateCritiqueAsync(scan);

            _context.ScanCritiques.Add(critique);
            await _context.SaveChangesAsync();

            return Ok(new
            {
                critique.Id,
                critique.ScanHistoryId,
                critique.ModelUsed,
                critique.CritiqueText,
                critique.Succeeded,
                critique.ErrorMessage,
                critique.GeneratedAt
            });
        }

        /// <summary>
        /// Fetches the most recent stored critique for a scan, without calling Gemini again.
        /// URL Endpoint: GET api/AdminDashboard/scans/{scanId}/critique
        /// </summary>
        [HttpGet("scans/{scanId}/critique")]
        public async Task<IActionResult> GetCritique(Guid scanId)
        {
            var critique = await _context.ScanCritiques
                .Where(c => c.ScanHistoryId == scanId)
                .OrderByDescending(c => c.GeneratedAt)
                .FirstOrDefaultAsync();

            if (critique == null)
                return NotFound(new { Message = "No critique has been generated for this scan yet." });

            return Ok(new
            {
                critique.Id,
                critique.ScanHistoryId,
                critique.ModelUsed,
                critique.CritiqueText,
                critique.Succeeded,
                critique.ErrorMessage,
                critique.GeneratedAt
            });
        }

        /// <summary>
        /// Soft-deletes a specific scan diagnostic profile from the administration feed.
        /// URL Endpoint: DELETE api/AdminDashboard/scans/{scanId}
        /// </summary>
        [HttpDelete("scans/{scanId}")]
        public async Task<IActionResult> SoftDeleteScan(Guid scanId)
        {
            // Note: Since global query filters are active, we use IgnoreQueryFilters() 
            // in case we need to locate a record that has already been toggled.
            var scan = await _context.ScanHistories
                .IgnoreQueryFilters()
                .FirstOrDefaultAsync(s => s.Id == scanId);

            if (scan == null)
            {
                return NotFound(new { Message = "The requested diagnostic record does not exist." });
            }

            if (scan.IsDeleted)
            {
                return BadRequest(new { Message = "This scan record has already been deleted." });
            }

            // Apply soft delete flag swap
            scan.IsDeleted = true;

            _context.ScanHistories.Update(scan);
            await _context.SaveChangesAsync();

            return Ok(new { Message = $"Scan record {scanId} soft-deleted successfully." });
        }
    }
}