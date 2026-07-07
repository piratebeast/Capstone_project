using System;
using System.Collections.Generic;
using System.IdentityModel.Tokens.Jwt;
using System.Linq;
using System.Security.Claims;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;
using SkincareAdvisor.Application.DTOs;
using SkincareAdvisor.Application.Interfaces;
using SkincareAdvisor.Domain.Entities;

namespace SkincareAdvisor.Infrastructure.Services
{
    /// <summary>
    /// Provides authentication operations for registering users, validating credentials,
    /// and issuing JWT tokens with full Role-Based Access Control (RBAC).
    /// </summary>
    public class AuthService : IAuthService
    {
        private readonly UserManager<ApplicationUser> _userManager;
        private readonly RoleManager<IdentityRole> _roleManager; // NEW: Injected to manage ASP.NET system roles
        private readonly IConfiguration _configuration;

        public AuthService(
            UserManager<ApplicationUser> userManager,
            RoleManager<IdentityRole> roleManager,
            IConfiguration configuration)
        {
            _userManager = userManager;
            _roleManager = roleManager;
            _configuration = configuration;
        }

        public async Task<AuthResponse> LoginAsync(LoginRequest request)
        {
            var user = await _userManager.FindByEmailAsync(request.Email);
            if (user == null)
            {
                return new AuthResponse(false, string.Empty, "Invalid email or password.");
            }

            var isPasswordValid = await _userManager.CheckPasswordAsync(user, request.Password);
            if (!isPasswordValid)
            {
                return new AuthResponse(false, string.Empty, "Invalid email or password.");
            }

            // MODIFIED: Generate token with roles asynchronously
            var token = await GenerateJwtTokenAsync(user);

            return new AuthResponse(true, token, "Login successful!");
        }

        public async Task<AuthResponse> RegisterAsync(RegisterRequest request)
        {
            var user = new ApplicationUser
            {
                UserName = request.Email,
                Email = request.Email,
                FullName = request.FullName,
                Gender = request.Gender,
                CreatedAt = DateTime.UtcNow,
                DateOfBirth = request.DateOfBirth
            };

            var result = await _userManager.CreateAsync(user, request.Password);
            if (!result.Succeeded)
            {
                var firstError = result.Errors.FirstOrDefault()?.Description ?? "Registration failed.";
                return new AuthResponse(false, string.Empty, firstError);
            }

            // ===================================================================
            // 🛠️ NEW LOGIC: DYNAMIC SCHEMATIC ROLE ASSIGNMENT
            // ===================================================================
            // Define standard application roles contract
            string targetRole = "Patient";

            // Capstone Testing Convenience Override: 
            // If the user registers with an administrative corporate email domain, seed them as an Admin!
            if (request.Email.EndsWith("@skinai.admin") || request.Email.Equals("admin@skinai.local"))
            {
                targetRole = "Admin";
            }

            // Ensure the role exists in the database table (AspNetRoles)
            if (!await _roleManager.RoleExistsAsync(targetRole))
            {
                await _roleManager.CreateAsync(new IdentityRole(targetRole));
            }

            // Map the user to the role in the junction mapping table (AspNetUserRoles)
            await _userManager.AddToRoleAsync(user, targetRole);
            // ===================================================================

            var token = await GenerateJwtTokenAsync(user);
            return new AuthResponse(true, token, "Registration successful!");
        }

        /// <summary>
        /// Creates a signed JWT token containing user identity attributes and role collections.
        /// </summary>
        private async Task<string> GenerateJwtTokenAsync(ApplicationUser user)
        {
            var jwtSettings = _configuration.GetSection("JwtSettings");
            var secretKey = jwtSettings["Key"];

            var symmetricSecurityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(secretKey!));
            var signingCredentials = new SigningCredentials(symmetricSecurityKey, SecurityAlgorithms.HmacSha256);

            var claims = new List<Claim>
            {
                new Claim(JwtRegisteredClaimNames.Sub, user.Id),
                new Claim(JwtRegisteredClaimNames.Email, user.Email!),
                new Claim("FullName", user.FullName ?? "")
            };

            // ===================================================================
            // 🛠️ NEW LOGIC: APPEND ROLES INTO THE JWT TOKEN CLAIMS STRINGS
            // ===================================================================
            // Fetch all roles assigned to this specific user from the database
            var userRoles = await _userManager.GetRolesAsync(user);
            foreach (var role in userRoles)
            {
                // ClaimTypes.Role maps directly to the standard C# [Authorize(Roles = "...")] tag analyzer
                claims.Add(new Claim(ClaimTypes.Role, role));
            }
            // ===================================================================

            var token = new JwtSecurityToken(
                issuer: jwtSettings["Issuer"],
                audience: jwtSettings["Audience"],
                claims: claims,
                expires: DateTime.UtcNow.AddMinutes(Convert.ToDouble(jwtSettings["DurationInMinutes"])),
                signingCredentials: signingCredentials
            );

            return new JwtSecurityTokenHandler().WriteToken(token);
        }
    }
}