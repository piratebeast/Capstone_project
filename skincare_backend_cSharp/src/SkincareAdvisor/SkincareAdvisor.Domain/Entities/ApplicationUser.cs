using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Identity;

namespace SkincareAdvisor.Domain.Entities
{
    public class ApplicationUser : IdentityUser
    {
        // IdentityUser already provides Id, Email, PasswordHash, and PhoneNumber.
        // Here we add your custom properties from your database design:
        public string? FullName { get; set; } = string.Empty;
        public string? AvatarUrl { get; set; } = string.Empty;
        public string? Gender { get; set; } = string.Empty;
        public DateTime? CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? DateOfBirth { get; set; }
    }
}
