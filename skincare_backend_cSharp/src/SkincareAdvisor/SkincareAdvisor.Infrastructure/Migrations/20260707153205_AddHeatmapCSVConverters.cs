using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SkincareAdvisor.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddHeatmapCSVConverters : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "AcneHeatmap",
                table: "ScanHistories",
                type: "VARCHAR(MAX)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "DarkCirclesHeatmap",
                table: "ScanHistories",
                type: "VARCHAR(MAX)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "DarkSpotsHeatmap",
                table: "ScanHistories",
                type: "VARCHAR(MAX)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "ImageUrl",
                table: "ScanHistories",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "RednessHeatmap",
                table: "ScanHistories",
                type: "VARCHAR(MAX)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "WrinklesHeatmap",
                table: "ScanHistories",
                type: "VARCHAR(MAX)",
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "AcneHeatmap",
                table: "ScanHistories");

            migrationBuilder.DropColumn(
                name: "DarkCirclesHeatmap",
                table: "ScanHistories");

            migrationBuilder.DropColumn(
                name: "DarkSpotsHeatmap",
                table: "ScanHistories");

            migrationBuilder.DropColumn(
                name: "ImageUrl",
                table: "ScanHistories");

            migrationBuilder.DropColumn(
                name: "RednessHeatmap",
                table: "ScanHistories");

            migrationBuilder.DropColumn(
                name: "WrinklesHeatmap",
                table: "ScanHistories");
        }
    }
}
