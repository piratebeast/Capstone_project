using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace SkincareAdvisor.Domain.Entities
{
    public class ScanHistory
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public string UserId { get; set; } = string.Empty;
        public DateTime ScanDate { get; set; } = DateTime.UtcNow;

        // --- BRAIN 1: CNN Outputs ---
        public double Acne { get; set; }
        public double DarkSpots { get; set; }
        public double Wrinkles { get; set; }
        public double Redness { get; set; }
        public double DarkCircles { get; set; }
        public string Gender { get; set; } = string.Empty;

        // --- BRAIN 2: Random Forest Outputs ---
        public string RoutineClass { get; set; } = string.Empty;
        public double Confidence { get; set; }

        // EF core will save these  as JSON columns automatically
        public List<RoutineStepEntity> DailyAm { get; set; } = new();
        public List<RoutineStepEntity> DailyPm { get; set; } = new();
        public List<WeeklyTreatmentEntity> WeeklyTreatments { get; set; } = new();
    }

    // Classes specifically for the Database
    public class RoutineStepEntity
    {
        public int Step { get; set; }
        public string Product { get; set; } = string.Empty;
        public string Purpose { get; set; } = string.Empty;
    }

    public class WeeklyTreatmentEntity
    {
        public string Product { get; set; } = string.Empty;
        public string Frequency { get; set; } = string.Empty;
        public string Instructions { get; set; } = string.Empty;
    }
}
