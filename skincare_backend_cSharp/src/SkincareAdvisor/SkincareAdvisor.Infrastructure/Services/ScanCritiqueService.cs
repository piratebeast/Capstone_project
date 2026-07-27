using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using SkincareAdvisor.Domain.Entities;
using SkincareAdvisor.Application.Interfaces;

namespace SkincareAdvisor.Infrastructure.Services
{
    public class ScanCritiqueService : IScanCritiqueService
    {
        private readonly HttpClient _http;
        private readonly string _apiKey;

        public ScanCritiqueService(HttpClient http, IConfiguration configuration)
        {
            _http = http;
            _apiKey = configuration["GEMINI_API_KEY"]
                ?? throw new ArgumentNullException(nameof(configuration), "GEMINI_API_KEY missing from configuration.");
        }

        public async Task<ScanCritique> GenerateCritiqueAsync(ScanHistory scan)
        {
            var critique = new ScanCritique
            {
                ScanHistoryId = scan.Id,
                ModelUsed = "gemini-3.1-flash-lite",
                GeneratedAt = DateTime.UtcNow
            };

            try
            {
                var physicalImagePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", scan.ImageUrl.TrimStart('/'));
                if (!File.Exists(physicalImagePath))
                    throw new FileNotFoundException($"Image not found at: {physicalImagePath}");

                var imageBytes = await File.ReadAllBytesAsync(physicalImagePath);
                var base64Image = Convert.ToBase64String(imageBytes);

                var promptText = $"""
                You are an expert dermatological AI audit assistant reviewing our custom ONNX model's output.
                Compare the visual reality of the attached face image against these predictions:
                - Routine Category: {scan.RoutineClass}
                - Overall Confidence: {scan.Confidence * 100:F2}%
                - Acne: {scan.Acne * 100:F2}%
                - Dark Spots/Hyperpigmentation: {scan.DarkSpots * 100:F2}%
                - Wrinkles/Fine Lines: {scan.Wrinkles * 100:F2}%
                - Redness/Erythema: {scan.Redness * 100:F2}%
                - Dark Circles: {scan.DarkCircles * 100:F2}%

                Determine if the model missed prominent conditions or over/under-diagnosed features.
                Respond in 2-3 concise, clinical sentences. Plain text, no markdown.
                """;

                var requestBody = new
                {
                    contents = new[]
                    {
                        new
                        {
                            parts = new object[]
                            {
                                new { text = promptText },
                                new { inline_data = new { mime_type = "image/jpeg", data = base64Image } }
                            }
                        }
                    }
                };

                //var url = $"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={_apiKey}";
                var url = $"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={_apiKey}";
                var response = await _http.PostAsJsonAsync(url, requestBody);
                var raw = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    critique.Succeeded = false;
                    critique.ErrorMessage = $"HTTP {response.StatusCode}: {raw}";
                    critique.CritiqueText = "AI Model Critique temporarily unavailable.";
                    critique.RawResponseJson = raw;
                    return critique;
                }

                using var doc = JsonDocument.Parse(raw);
                var text = doc.RootElement
                    .GetProperty("candidates")[0]
                    .GetProperty("content")
                    .GetProperty("parts")[0]
                    .GetProperty("text")
                    .GetString() ?? "No critique text returned.";

                critique.CritiqueText = text;
                critique.Succeeded = true;
                critique.RawResponseJson = raw;
            }
            catch (Exception ex)
            {
                critique.Succeeded = false;
                critique.ErrorMessage = ex.Message;
                critique.CritiqueText = "AI Model Critique temporarily unavailable due to upstream failure.";
            }

            return critique;
        }
    }
}