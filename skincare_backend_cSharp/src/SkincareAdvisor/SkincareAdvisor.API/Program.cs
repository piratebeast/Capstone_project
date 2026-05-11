using System.Text;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using SkincareAdvisor.Application.Interfaces;
using SkincareAdvisor.Domain.Entities;
using SkincareAdvisor.Infrastructure.Persistence;
using SkincareAdvisor.Infrastructure.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

// 1. Setup Database Connection
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

// 2. Setup ASP.NET Identity
builder.Services.AddIdentity<ApplicationUser, IdentityRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders();

// 3. Setup JWT Authentication Service 
// This pulls the key from your environment variables or appsettings
var jwtKey = builder.Configuration["JwtSettings:Key"]
             ?? builder.Configuration["JwtSettings__Key"];

if (string.IsNullOrEmpty(jwtKey))
{
    // If this hits, double-check your launchSettings.json for "JwtSettings__Key"
    throw new Exception("CRITICAL ERROR: JWT Secret Key is missing from configuration.");
}

builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuer = false, // Set to true if you have a specific issuer in appsettings
        ValidateAudience = false, // Set to true if you have a specific audience in appsettings
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey))
    };
});

builder.Services.AddControllers();

// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

//4. Dependency Injection
builder.Services.AddScoped<IAuthService, AuthService>();

// This handles the HttpClient pooling automatically!
builder.Services.AddHttpClient<IScanService, ScanService>();

//Api Rates Limiting (Optional, but recommended for production)
builder.Services.AddRateLimiter(options =>
{
    options.AddFixedWindowLimiter("api-policy", opt =>
    {
        opt.Window = TimeSpan.FromMinutes(1);
        opt.PermitLimit = 10; // Allow 10 scans per minute
        opt.QueueLimit = 2;   // Queue 2 extra if they are just slightly over
    });
});


var app = builder.Build();


// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

// 5. Middleware Order (CRITICAL)
app.UseRouting();

// Apply the rate limiter BEFORE authentication and authorization to prevent abuse of those systems
app.UseRateLimiter();

// Authentication MUST come before Authorization
app.UseAuthentication(); // Checks the JWT token
app.UseAuthorization();  // Decides if the user can access the endpoint

app.MapControllers();

app.Run();