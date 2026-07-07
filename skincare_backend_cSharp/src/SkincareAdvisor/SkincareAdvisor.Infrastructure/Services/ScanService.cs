using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using SkincareAdvisor.Application.DTOs;
using SkincareAdvisor.Application.Interfaces;

namespace SkincareAdvisor.Infrastructure.Services
{
    public class ScanService : IScanService
    {
        private readonly HttpClient _httpClient;

        public ScanService(HttpClient httpClient)
        {
            _httpClient = httpClient;

            // This is the default port for FastAPI. Change it if your Python runs elsewhere!
            _httpClient.BaseAddress = new Uri("http://localhost:8000/");
        }

        public async Task<AiScanResponse> AnalyzeImageAsync(IFormFile image, int userAge)
        {
            using var content = new MultipartFormDataContent();

            // Pack the image file
            using var stream = image.OpenReadStream();
            content.Add(new StreamContent(stream), "file", image.FileName);

            // Pack the user's age
            content.Add(new StringContent(userAge.ToString()), "user_age");

            // Send both the image and the age to Python
            var response = await _httpClient.PostAsync("analyze", content);

            // --- Check if Python rejected the image! ---
            if (!response.IsSuccessStatusCode)
            {
                var errorJson = await response.Content.ReadAsStringAsync();

                if (response.StatusCode == System.Net.HttpStatusCode.BadRequest && errorJson.Contains("NO_FACE_DETECTED"))
                {
                    throw new ArgumentException("No human face detected in the image.");
                }
                if (response.StatusCode == System.Net.HttpStatusCode.BadRequest && errorJson.Contains("MULTIPLE_FACES_DETECTED"))
                {
                    throw new ArgumentException("Multiple faces detected. Please upload a solo selfie.");
                }

                throw new Exception($"The AI Service encountered an error. Status: {response.StatusCode}");
            }

            // ===================================================================
            // 🛠️ MODIFICATION: Configure case-insensitive mapping options
            // ===================================================================
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            // Deserialize using our optimized options layout to ingest all 5 float list vectors safely
            var result = await response.Content.ReadFromJsonAsync<AiScanResponse>(jsonOptions);

            if (result == null)
            {
                throw new Exception("Failed to deserialize the AI analysis payload payload contract.");
            }

            return result;
        }
    }
}