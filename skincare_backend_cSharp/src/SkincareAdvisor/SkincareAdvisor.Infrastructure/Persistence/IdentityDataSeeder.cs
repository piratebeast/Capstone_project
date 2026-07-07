using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.DependencyInjection;
using SkincareAdvisor.Domain.Entities;

namespace SkincareAdvisor.Infrastructure.Persistence
{
    public static class IdentityDataSeeder
    {
        public static async Task SeedAdminUserAsync(IServiceProvider serviceProvider)
        {
            // Resolve the core Identity managers from the application's service container
            var userManager = serviceProvider.GetRequiredService<UserManager<ApplicationUser>>();
            var roleManager = serviceProvider.GetRequiredService<RoleManager<IdentityRole>>();

            string adminRole = "Admin";
            string adminEmail = "admin@skinai.local";
            string adminPassword = "SecureAdminPassword123!"; // Must meet default Identity complexity guidelines

            // 1. Ensure the "Admin" role exists in the database
            if (!await roleManager.RoleExistsAsync(adminRole))
            {
                await roleManager.CreateAsync(new IdentityRole(adminRole));
            }

            // 2. Ensure a global standard "Patient" role exists as well
            if (!await roleManager.RoleExistsAsync("Patient"))
            {
                await roleManager.CreateAsync(new IdentityRole("Patient"));
            }

            // 3. Check if our target administrative system profile already exists
            var existingAdmin = await userManager.FindByEmailAsync(adminEmail);
            if (existingAdmin == null)
            {
                // Instantiate the identity entity model mapping parameters
                var adminUser = new ApplicationUser
                {
                    UserName = adminEmail,
                    Email = adminEmail,
                    FullName = "System Administrator",
                    Gender = "System",
                    CreatedAt = DateTime.UtcNow,
                    DateOfBirth = new DateTime(2000, 1, 1),
                    EmailConfirmed = true // Bypasses verification flags automatically
                };

                // CreateAsync automatically hashes and salts the password behind the scenes
                var createResult = await userManager.CreateAsync(adminUser, adminPassword);

                if (createResult.Succeeded)
                {
                    // Assign the created entity profile directly into the Admin security group
                    await userManager.AddToRoleAsync(adminUser, adminRole);
                    Console.WriteLine("✅ SEED DATA: Administrative profile created successfully!");
                }
                else
                {
                    var errorMsg = string.Join(", ", createResult.Errors.Select(e => e.Description));
                    Console.WriteLine($"❌ SEED DATA ERROR: Failed to create admin user. Errors: {errorMsg}");
                }
            }
            else
            {
                Console.WriteLine("ℹ️ SEED DATA: Administrative profile already exists in database context.");
            }
        }
    }
}