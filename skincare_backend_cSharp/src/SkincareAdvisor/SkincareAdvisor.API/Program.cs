using System.Security.Claims; // <-- NEW: Explicitly needed for ClaimTypes.Role mapping
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

// 2. Setup ASP.NET Identity (Now using RoleManager configurations behind the scenes)
builder.Services.AddIdentity<ApplicationUser, IdentityRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders();

// 3. Setup JWT Authentication Service 
var jwtKey = builder.Configuration["JwtSettings:Key"]
             ?? builder.Configuration["JwtSettings__Key"];

if (string.IsNullOrEmpty(jwtKey))
{
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
        ValidateIssuer = false,
        ValidateAudience = false,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey)),

        // ===================================================================
        // CRITICAL ADDITION FOR ADMIN PANEL SECURITY
        // ===================================================================
        // Maps incoming identity role schema attributes to the [Authorize(Roles = "Admin")] analyzer
        RoleClaimType = ClaimTypes.Role
        // ===================================================================
    };
});

builder.Services.AddCors();
builder.Services.AddControllers();

// Configure Swagger/OpenAPI
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// 4. Dependency Injection Mapping
builder.Services.AddScoped<IAuthService, AuthService>();

// Handles the HttpClient pooling automatically for your FastAPI connector!
builder.Services.AddHttpClient<IScanService, ScanService>();

// Api Rates Limiting Configuration
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

// ===================================================================
// NEW: EXECUTE IDENTITY INITIALIZATION DATA SEEDING AT STARTUP
// ===================================================================
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    try
    {
        // Invoke our static manager mapping loop asynchronously
        await SkincareAdvisor.Infrastructure.Persistence.IdentityDataSeeder.SeedAdminUserAsync(services);
    }
    catch (Exception ex)
    {
        var logger = services.GetRequiredService<ILogger<Program>>();
        logger.LogError(ex, "An initialization error struck the database migration seed system.");
    }
}
// ===================================================================

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseStaticFiles(new StaticFileOptions
{
    OnPrepareResponse = ctx =>
    {
        ctx.Context.Response.Headers.Append("Access-Control-Allow-Origin", "*");
        ctx.Context.Response.Headers.Append("Access-Control-Allow-Headers", "*");
        ctx.Context.Response.Headers.Append("Access-Control-Allow-Methods", "*");
    }
});

// 5. Middleware Order (CRITICAL)
app.UseRouting();
app.UseCors(policy => policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod());

// Apply the rate limiter BEFORE authentication and authorization to prevent system abuse
app.UseRateLimiter();

// Authentication MUST come before Authorization
app.UseAuthentication(); // Decodes the JWT token passport and extracts claims/roles
app.UseAuthorization();  // Decides if the active token role possesses permission targets

app.MapControllers();

app.Run();