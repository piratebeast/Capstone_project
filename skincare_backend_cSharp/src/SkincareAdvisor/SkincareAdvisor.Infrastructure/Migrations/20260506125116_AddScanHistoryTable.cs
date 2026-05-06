using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SkincareAdvisor.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddScanHistoryTable : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ScanHistory",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    UserId = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    ScanDate = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Acne = table.Column<double>(type: "float", nullable: false),
                    DarkSpots = table.Column<double>(type: "float", nullable: false),
                    Wrinkles = table.Column<double>(type: "float", nullable: false),
                    Redness = table.Column<double>(type: "float", nullable: false),
                    DarkCircles = table.Column<double>(type: "float", nullable: false),
                    Gender = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    RoutineClass = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Confidence = table.Column<double>(type: "float", nullable: false),
                    DailyAm = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    DailyPm = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    WeeklyTreatments = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ScanHistory", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "ScanHistory");
        }
    }
}
