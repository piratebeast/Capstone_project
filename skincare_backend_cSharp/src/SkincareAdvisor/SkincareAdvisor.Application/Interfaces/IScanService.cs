using Microsoft.AspNetCore.Http;
using SkincareAdvisor.Application.DTOs;
using System.Threading.Tasks;

namespace SkincareAdvisor.Application.Interfaces
{
    public interface IScanService
    {
        // Add "int userAge" here
        Task<AiScanResponse> AnalyzeImageAsync(IFormFile image, int userAge);
    }
}