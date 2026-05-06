using System;
using System.Net.Http;
using System.Net.Http.Headers;
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
            // 1. Create the multipart form data content
            using var content = new MultipartFormDataContent();
            using var stream = image.OpenReadStream();
            var fileContent = new StreamContent(stream);
            fileContent.Headers.ContentType = new MediaTypeHeaderValue(image.ContentType);

            // NOTE: "file" must match whatever your Python FastAPI endpoint expects as the parameter name!
            content.Add(fileContent, "file", image.FileName);

            // 2. Send the POST request to the FastAPI endpoint
            var response = await _httpClient.PostAsync("predict", content);

            // 3. Ensure if python crashed
            if (!response.IsSuccessStatusCode) 
            {
                var errorMsg = await response.Content.ReadAsStringAsync();
                throw new Exception($"Python API failed with status {response.StatusCode}: {errorMsg}");
            }

            // 4. Read the JSON and magically convert it into your C# DTO
            var jsonString = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<AiScanResponse>(jsonString);

            return result ?? throw new Exception("Failed to deserialize Python response.");
        }
    }
}
