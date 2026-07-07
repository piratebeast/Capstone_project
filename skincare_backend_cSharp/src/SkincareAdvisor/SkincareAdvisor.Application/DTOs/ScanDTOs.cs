using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Http;
using System.Collections.Generic;

namespace SkincareAdvisor.Application.DTOs
{
    // What Flutter sends to C#
    public record ScanRequest(IFormFile image);

    // The Massive Combined JSON Python sends back to C# (Now with Heatmaps!)
    public record AiScanResponse(
        [property: JsonPropertyName("diagnostics")] DiagnosticsDto Diagnostics,
        [property: JsonPropertyName("heatmaps")] HeatmapsDto Heatmaps, // <-- NEW: Added to map flat activation metrics
        [property: JsonPropertyName("routineClass")] string RoutineClass,
        [property: JsonPropertyName("confidence")] double Confidence,
        [property: JsonPropertyName("regimenSchedule")] RegimenScheduleDto RegimenSchedule
    );

    // --- BRAIN 1: CNN Outputs ---
    public record DiagnosticsDto(
        [property: JsonPropertyName("acne")] double Acne,
        [property: JsonPropertyName("dark_spots")] double DarkSpots,
        [property: JsonPropertyName("wrinkles")] double Wrinkles,
        [property: JsonPropertyName("redness")] double Redness,
        [property: JsonPropertyName("dark_circles")] double DarkCircles,
        [property: JsonPropertyName("gender")] string Gender
    );

    // --- NEW: Map Grid Activation Nodes (50,176 elements per list) ---
    public record HeatmapsDto(
        [property: JsonPropertyName("acne")] List<float> Acne,
        [property: JsonPropertyName("darkSpots")] List<float> DarkSpots,   // Matches Python's camelCase dictionary key
        [property: JsonPropertyName("wrinkles")] List<float> Wrinkles,
        [property: JsonPropertyName("redness")] List<float> Redness,
        [property: JsonPropertyName("darkCircles")] List<float> DarkCircles // Matches Python's camelCase dictionary key
    );

    // --- BRAIN 2: Random Forest Outputs ---
    public record RegimenScheduleDto(
        [property: JsonPropertyName("dailyAm")] List<RoutineStepDto> DailyAm,
        [property: JsonPropertyName("dailyPm")] List<RoutineStepDto> DailyPm,
        [property: JsonPropertyName("weeklyTreatments")] List<WeeklyTreatmentDto> WeeklyTreatments
    );

    public record RoutineStepDto(
        [property: JsonPropertyName("step")] int Step,
        [property: JsonPropertyName("product")] string Product,
        [property: JsonPropertyName("purpose")] string Purpose
    );

    public record WeeklyTreatmentDto(
        [property: JsonPropertyName("product")] string Product,
        [property: JsonPropertyName("frequency")] string Frequency,
        [property: JsonPropertyName("instructions")] string Instructions
    );
}   