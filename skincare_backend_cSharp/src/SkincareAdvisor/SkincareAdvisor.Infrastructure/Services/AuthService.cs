using System.IdentityModel.Tokens.Jwt;
using System.Linq;
using System.Security.Claims;
using System.Text;
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
    /// and issuing JWT tokens.
    /// </summary>
    public class AuthService : IAuthService
    {
        private readonly UserManager<ApplicationUser> _userManager;
        private readonly IConfiguration _configuration;

        //Dependency Injection: We inject UserManager and IConfiguration to handle user operations and access app settings.
        public AuthService(UserManager<ApplicationUser> userManager, IConfiguration configuration)
        {
            _userManager = userManager;
            _configuration = configuration;
        }

        /// <summary>
        /// Validates user credentials and returns a JWT token when authentication succeeds.
        /// </summary>
        /// <param name="request">The login request containing email and password.</param>
        /// <returns>An authentication response containing a token on success.</returns>
        public async Task<AuthResponse> LoginAsync(LoginRequest request)
        {
            // 1. Use _userManager to find the user by their email
            // This goes to the AspNetUsers table in your SQL Server
            var user = await _userManager.FindByEmailAsync(request.Email);

            // 2. If the user doesn't exist, stop here.
            // For security, we use a generic message so hackers don't know which part was wrong.
            if (user == null)
            {
                return new AuthResponse(false, string.Empty, "Invalid email or password.");
            }

            // 3. Use _userManager to check if the password is correct
            // Identity handles the hashing and salting comparison automatically
            var isPasswordValid = await _userManager.CheckPasswordAsync(user, request.Password);

            // 4. If the password is wrong, stop here with the same generic message.
            if (!isPasswordValid)
            {
                return new AuthResponse(false, string.Empty, "Invalid email or password.");
            }

            // 5. If everything is correct, generate the digital "passport" (JWT)
            var token = GenerateJwtToken(user);

            // 6. Return the token to the client so they can use it for future requests
            return new AuthResponse(true, token, "Login successful!");
        }

        /// <summary>
        /// Registers a new user and returns a JWT token when registration succeeds.
        /// </summary>
        /// <param name="request">The registration request containing user details.</param>
        /// <returns>An authentication response containing a token on success.</returns>
        public async Task<AuthResponse> RegisterAsync(RegisterRequest request)
        {
            // 1. Map DTO to Entity
            var user = new ApplicationUser
            {
                UserName = request.Email, // Identity uses UserName for unique identification
                Email = request.Email,
                FullName = request.FullName,
                Gender = request.Gender,
                CreatedAt = DateTime.UtcNow,
                DateOfBirth = request.DateOfBirth
            };

            // 2. Attempt to create the user
            // Note: CreateAsync returns an 'IdentityResult' object
            var result = await _userManager.CreateAsync(user, request.Password);

            // 3. Handle failure
            if (!result.Succeeded)
            {
                // Logic Tip: result.Errors is a collection. 
                // Grab the first description to show the user what went wrong.
                var firstError = result.Errors.FirstOrDefault()?.Description ?? "Registration failed.";
                return new AuthResponse(false, string.Empty, firstError);
            }

            // 4. Handle success
            // Call the private helper method we wrote earlier
            var token = GenerateJwtToken(user);

            return new AuthResponse(true, token, "Registration successful!");
        }

        // --- BOILERPLATE: JWT Generation Logic ---
        /// <summary>
        /// Creates a signed JWT token for the specified user using configuration settings.
        /// </summary>
        /// <param name="user">The authenticated user.</param>
        /// <returns>A serialized JWT token.</returns>
        private string GenerateJwtToken(ApplicationUser user)
        {
            var jwtSettings = _configuration.GetSection("JwtSettings");
            var secretKey = jwtSettings["Key"];

            // 1. Create the security key from your environment variable
            var symmetricSecurityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(secretKey!));
            var signingCredentials = new SigningCredentials(symmetricSecurityKey, SecurityAlgorithms.HmacSha256);

            // 2. Define the "Claims" (the data hidden inside the token)
            var claims = new List<Claim>
        {
            new Claim(JwtRegisteredClaimNames.Sub, user.Id),
            new Claim(JwtRegisteredClaimNames.Email, user.Email!),
            new Claim("FullName", user.FullName ?? "") // Custom claim for your app
        };

            // 3. Assemble the token
            var token = new JwtSecurityToken(
                issuer: jwtSettings["Issuer"],
                audience: jwtSettings["Audience"],
                claims: claims,
                expires: DateTime.UtcNow.AddMinutes(Convert.ToDouble(jwtSettings["DurationInMinutes"])),
                signingCredentials: signingCredentials
            );

            // 4. Serialize the token to a string so Flutter can read it
            return new JwtSecurityTokenHandler().WriteToken(token);
        }

    }
}
