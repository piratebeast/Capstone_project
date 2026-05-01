using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using SkincareAdvisor.Domain.Entities;

namespace SkincareAdvisor.Infrastructure.Persistence
{
    /// <summary>
    /// Represents the Entity Framework Core database context for the Skincare Advisor application.
    /// Inherits from IdentityDbContext to provide built-in Identity management capabilities for
    /// user authentication and authorization.
    /// This context handles all database operations and entity mapping for the application's data model.
    /// </summary>
    public class ApplicationDbContext : IdentityDbContext<ApplicationUser>
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {

        }

        protected override void OnModelCreating(ModelBuilder builder)
        {
            base.OnModelCreating(builder);

            // We can add custom database table configurations here later if needed
        }
    }
}
