using DotNetEnv;
using System.Security.Claims;
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

Env.Load(); // load .env FIRST, before builder reads config

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddEnvironmentVariables();

// 1. Database
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

// 2. Identity
builder.Services.AddIdentity<ApplicationUser, IdentityRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders();

// 3. JWT
var jwtKey = builder.Configuration["JwtSettings:Key"]
             ?? builder.Configuration["JwtSettings__Key"];

if (string.IsNullOrEmpty(jwtKey))
    throw new Exception("CRITICAL ERROR: JWT Secret Key is missing from configuration.");

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
        RoleClaimType = ClaimTypes.Role
    };
});

builder.Services.AddCors();
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// 4. DI registrations (FIXED: Registered cleanly as Scoped instead of HttpClient wrapper)
builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddHttpClient<IScanCritiqueService, ScanCritiqueService>();
builder.Services.AddHttpClient<IScanService, ScanService>();

// 5. Rate limiting
builder.Services.AddRateLimiter(options =>
{
    options.AddFixedWindowLimiter("api-policy", opt =>
    {
        opt.Window = TimeSpan.FromMinutes(1);
        opt.PermitLimit = 10;
        opt.QueueLimit = 2;
    });
});

var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    try
    {
        await SkincareAdvisor.Infrastructure.Persistence.IdentityDataSeeder.SeedAdminUserAsync(services);
    }
    catch (Exception ex)
    {
        var logger = services.GetRequiredService<ILogger<Program>>();
        logger.LogError(ex, "An initialization error struck the database migration seed system.");
    }
}

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

app.UseRouting();
app.UseCors(policy => policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod());
app.UseRateLimiter();
app.UseAuthentication();
app.UseAuthorization();
app.MapControllers();
app.Run();