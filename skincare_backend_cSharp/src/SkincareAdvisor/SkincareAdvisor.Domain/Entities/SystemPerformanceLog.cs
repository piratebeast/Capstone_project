using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SkincareAdvisor.Domain.Entities
{
    public class SystemPerformanceLog
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public string? UserId { get; set; } // Optional: track who ran it, null for unauthenticated failures
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;

        // Performance Telemetry
        public int InferenceLatencyMs { get; set; } // Tracks how long Python took
        public bool IsSuccess { get; set; } // True if complete scan, False if MediaPipe failed

        // Error Classification for your "MediaPipe Fail-Fast" Widget
        public string? FailureReason { get; set; } // "NO_FACE_DETECTED", "MULTIPLE_FACES_DETECTED", "SERVER_ERROR", or null

        // Route Tracker for system statistics
        public string Endpoint { get; set; } = "api/Scan/analyze";
    }
}
