using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SkincareAdvisor.Infrastructure.Persistence;

namespace SkincareAdvisor.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    // Lock this entire controller down to identities possessing the verified Admin token claim
    [Authorize(Roles = "Admin")]
    public class AdminDashboardController : ControllerBase
    {
        private readonly ApplicationDbContext _context;

        public AdminDashboardController(ApplicationDbContext context)
        {
            _context = context;
        }

        /// <summary>
        /// Fetches an expansive data transaction profile for your Angular Live Diagnostic Feed dashboard.
        /// URL Endpoint: GET api/AdminDashboard/scans/{scanId}
        /// </summary>
        [HttpGet("scans/{scanId}")]
        public async Task<IActionResult> GetScanDetailForAdmin(Guid scanId)
        {
            // Pull the exact scan record from SQL Server
            var scan = await _context.ScanHistories
                .FirstOrDefaultAsync(s => s.Id == scanId);

            if (scan == null)
            {
                return NotFound(new { Message = "The requested diagnostic record does not exist." });
            }

            // Return the complete mapping properties contract directly down the wire to Angular
            return Ok(new
            {
                ScanId = scan.Id,
                UserId = scan.UserId,
                ScanDate = scan.ScanDate,
                RoutineClass = scan.RoutineClass,
                Confidence = scan.Confidence,
                OriginalImageUrl = scan.ImageUrl, // Web relative URL file path for Layer 1 drawing
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
                    // EF Core custom Value Converters automatically split the strings back into float lists here!
                    Acne = scan.AcneHeatmap,        // Flat array of 50,176 elements
                    DarkSpots = scan.DarkSpotsHeatmap,  // Flat array of 50,176 elements
                    Wrinkles = scan.WrinklesHeatmap,    // Flat array of 50,176 elements
                    Redness = scan.RednessHeatmap,      // Flat array of 50,176 elements
                    DarkCircles = scan.DarkCirclesHeatmap // Flat array of 50,176 elements
                }
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
                    PrimaryConditionSeverity = s.Acne // Easily extendable for reporting tables
                })
                .ToListAsync();

            return Ok(summaryList);
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