using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SkincareAdvisor.Application.DTOs
{

        public record RegisterRequest(
            string Email,
            string Password,
            string FullName,
            string Gender
            );

        public record LoginRequest(
            string Email,
            string Password
            );

        public record AuthResponse(
            bool Success,
            string token,
            string Message
            );
}
