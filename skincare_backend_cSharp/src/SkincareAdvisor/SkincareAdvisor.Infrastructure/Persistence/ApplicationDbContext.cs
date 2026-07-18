using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion; // <-- NEW: Required for explicit ValueConverter class mapping
using SkincareAdvisor.Domain.Entities;

namespace SkincareAdvisor.Infrastructure.Persistence
{
    /// <summary>
    /// Represents the Entity Framework Core database context for the Skincare Advisor application.
    /// Handles all database operations, built-in Identity capabilities, and custom value serialization.
    /// </summary>
    public class ApplicationDbContext : IdentityDbContext<ApplicationUser>
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }

        public DbSet<ScanHistory> ScanHistories { get; set; }
        public DbSet<ScanCritique> ScanCritiques { get; set; }
        public DbSet<SystemPerformanceLog> SystemPerformanceLogs { get; set; }

        protected override void OnModelCreating(ModelBuilder builder)
        {
            // Always call base first to seed Identity schemas
            base.OnModelCreating(builder);

            // 1. Existing JSON column configurations for regimen arrays
            builder.Entity<ScanHistory>().OwnsMany(s => s.DailyAm, a => a.ToJson());
            builder.Entity<ScanHistory>().OwnsMany(s => s.DailyPm, p => p.ToJson());
            builder.Entity<ScanHistory>().OwnsMany(s => s.WeeklyTreatments, w => w.ToJson());

            // 2. Configure High-Performance Comma-Separated Value Converters for Heatmaps
            ConfigureHeatmapConverter(builder, nameof(ScanHistory.AcneHeatmap));
            ConfigureHeatmapConverter(builder, nameof(ScanHistory.DarkSpotsHeatmap));
            ConfigureHeatmapConverter(builder, nameof(ScanHistory.WrinklesHeatmap));
            ConfigureHeatmapConverter(builder, nameof(ScanHistory.RednessHeatmap));
            ConfigureHeatmapConverter(builder, nameof(ScanHistory.DarkCirclesHeatmap));

            // Global filter - automatically excludes soft-deleted rows from all queries
            builder.Entity<ScanHistory>().HasQueryFilter(s => !s.IsDeleted);

            // Matching filter so ScanCritique respects its parent ScanHistory's soft-delete state
            builder.Entity<ScanCritique>().HasQueryFilter(c => !c.ScanHistory!.IsDeleted);
        }

        /// <summary>
        /// Binds a List of floats to a lightweight comma-delimited string layout within SQL Server.
        /// Uses an explicit ValueConverter instance to fix method signature mapping overloads.
        /// </summary>
        private void ConfigureHeatmapConverter(ModelBuilder builder, string propertyName)
        {
            // 1. Explicitly define the Value Converter to map List<float> <-> string
            var heatmapConverter = new ValueConverter<List<float>, string>(
                // To Database (Convert List<float> -> "0.0,0.412,0.92...")
                v => string.Join(',', v),

                // From Database (Split string -> List<float>)
                v => string.IsNullOrEmpty(v)
                    ? new List<float>()
                    : v.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(float.Parse).ToList()
            );

            // 2. Create a custom tracking comparer so EF Core can track mutations inside the list
            var heatmapValueComparer = new ValueComparer<List<float>>(
                (c1, c2) => c1 != null && c2 != null ? c1.SequenceEqual(c2) : c1 == c2,
                c => c.Aggregate(0, (a, v) => HashCode.Combine(a, v.GetHashCode())),
                c => c.ToList()
            );

            // 3. Attach the explicitly typed elements to the EF Core Fluent Configuration pipeline
            builder.Entity<ScanHistory>()
                .Property(propertyName)
                .HasConversion(heatmapConverter) // <-- Pass the converter object instance directly!
                .Metadata.SetValueComparer(heatmapValueComparer);

            // 4. Ensure optimal database text allocation storage bounds
            builder.Entity<ScanHistory>()
                .Property(propertyName)
                .HasColumnType("VARCHAR(MAX)");
        }
    }
}