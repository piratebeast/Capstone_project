using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Http;
using System.Collections.Generic; // Added just in case List<> throws an error

namespace SkincareAdvisor.Application.DTOs
{
    // What Flutter sends to C#
    public record ScanRequest(IFormFile image);

    // The Massive Combined JSON Python sends back to C#
    public record AiScanResponse(
        [property: JsonPropertyName("diagnostics")] DiagnosticsDto Diagnostics,
        [property: JsonPropertyName("routineClass")] string RoutineClass, // <-- CHANGED to camelCase
        [property: JsonPropertyName("confidence")] double Confidence,
        [property: JsonPropertyName("regimenSchedule")] RegimenScheduleDto RegimenSchedule // <-- CHANGED to camelCase
    );

    // --- BRAIN 1: CNN Outputs ---
    public record DiagnosticsDto(
        [property: JsonPropertyName("acne")] double Acne,
        [property: JsonPropertyName("dark_spots")] double DarkSpots, // <-- Kept snake_case based on your JSON
        [property: JsonPropertyName("wrinkles")] double Wrinkles,
        [property: JsonPropertyName("redness")] double Redness,
        [property: JsonPropertyName("dark_circles")] double DarkCircles, // <-- Kept snake_case based on your JSON
        [property: JsonPropertyName("gender")] string Gender
    );

    // --- BRAIN 2: Random Forest Outputs ---
    public record RegimenScheduleDto(
        [property: JsonPropertyName("dailyAm")] List<RoutineStepDto> DailyAm, // <-- CHANGED to camelCase
        [property: JsonPropertyName("dailyPm")] List<RoutineStepDto> DailyPm, // <-- CHANGED to camelCase
        [property: JsonPropertyName("weeklyTreatments")] List<WeeklyTreatmentDto> WeeklyTreatments // <-- CHANGED to camelCase
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