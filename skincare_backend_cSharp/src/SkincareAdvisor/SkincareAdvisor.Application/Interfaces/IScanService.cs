using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using SkincareAdvisor.Application.DTOs;

namespace SkincareAdvisor.Application.Interfaces
{
    public interface IScanService
    {
        // Takes the image, talks to Python, returns the massive JSON DTO
        Task<AiScanResponse> AnalyzeImageAsync(IFormFile image);
    }
}
