using Microsoft.AspNetCore.Mvc;
using SkincareAdvisor.Application.DTOs;
using SkincareAdvisor.Application.Interfaces;

namespace SkincareAdvisor.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AuthController : ControllerBase
    {
        private readonly IAuthService _authService;

        // Dependency Injection: We inject the IAuthService to handle authentication logic.
        public AuthController(IAuthService authService)
        {
            _authService = authService;
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            // 1. Call the service
            var response = await _authService.RegisterAsync(request);

            // 2. If it failed, return a 400 Bad Request with the error message
            if (!response.Success)
            {
                return BadRequest(response);
            }

            // 3. If it succeeded, return a 200 OK with the token!
            return Ok(response);
        }

        [HttpPost("login")]
        public async Task<IActionResult> Login([FromBody] LoginRequest request)
        {

            // 1. Call the service
            var response = await _authService.LoginAsync(request);

            // 2. If it failed, return a 400 Bad Request with the error message
            if (!response.Success) 
            {
                return BadRequest(response);
            }

            // 3. If it succeeded, return a 200 OK with the token!
            return Ok(response);
        }

    }
}
