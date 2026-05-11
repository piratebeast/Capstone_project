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

        public async Task<AiScanResponse> AnalyzeImageAsync(IFormFile image)
        {
            using var content = new MultipartFormDataContent();
            using var stream = image.OpenReadStream();
            content.Add(new StreamContent(stream), "file", image.FileName);

            var response = await _httpClient.PostAsync("http://localhost:8000/analyze", content);

            // --- NEW LOGIC: Check if Python rejected the image! ---
            if (!response.IsSuccessStatusCode)
            {
                var errorJson = await response.Content.ReadAsStringAsync();

                // If Python didn't find a face, throw a specific error
                if (response.StatusCode == System.Net.HttpStatusCode.BadRequest && errorJson.Contains("NO_FACE_DETECTED"))
                {
                    throw new ArgumentException("No human face detected in the image.");
                }
                if (response.StatusCode == System.Net.HttpStatusCode.BadRequest && errorJson.Contains("MULTIPLE_FACES_DETECTED"))
                {
                    throw new ArgumentException("Multiple faces detected. Please upload a solo selfie.");
                }

                // Generic fallback for any other Python crashes
                throw new Exception("The AI Service encountered an error.");
            }
            // ------------------------------------------------------

            // If successful (200 OK), deserialize it normally
            var result = await response.Content.ReadFromJsonAsync<AiScanResponse>();
            return result;
        }
    }
}
