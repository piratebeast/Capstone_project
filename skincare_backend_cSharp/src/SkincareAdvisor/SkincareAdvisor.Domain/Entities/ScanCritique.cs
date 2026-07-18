using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SkincareAdvisor.Domain.Entities
{
    public class ScanCritique
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public Guid ScanHistoryId { get; set; }        // FK
        public ScanHistory ScanHistory { get; set; } = null!;
        public string ModelUsed { get; set; } = "gemini-2.5-flash";
        public string CritiqueText { get; set; } = string.Empty;   // the audit text
        public string? RawResponseJson { get; set; }               // for debugging/audit
        public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
        public bool Succeeded { get; set; } = true;
        public string? ErrorMessage { get; set; }                  // populate on Gemini failure
    }
}
