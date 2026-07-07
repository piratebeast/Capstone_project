using System.Threading.Tasks;
using SkincareAdvisor.Application.DTOs;

namespace SkincareAdvisor.Application.Interfaces
{
    public interface IAuthService
    {
        // The 'Task' keyword means these methods will run asynchronously 
        // so they don't freeze your API while waiting for the database.

        Task<AuthResponse> RegisterAsync(RegisterRequest request);
        Task<AuthResponse> LoginAsync(LoginRequest request);
    }
}
