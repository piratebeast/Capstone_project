using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SkincareAdvisor.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddSoftDeleteToScanHistory : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "ScanHistories",
                type: "bit",
                nullable: false,
                defaultValue: false);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "ScanHistories");
        }
    }
}
