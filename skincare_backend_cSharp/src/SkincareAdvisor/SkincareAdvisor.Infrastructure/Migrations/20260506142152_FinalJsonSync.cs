using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SkincareAdvisor.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class FinalJsonSync : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropPrimaryKey(
                name: "PK_ScanHistory",
                table: "ScanHistory");

            migrationBuilder.RenameTable(
                name: "ScanHistory",
                newName: "ScanHistories");

            migrationBuilder.AddPrimaryKey(
                name: "PK_ScanHistories",
                table: "ScanHistories",
                column: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropPrimaryKey(
                name: "PK_ScanHistories",
                table: "ScanHistories");

            migrationBuilder.RenameTable(
                name: "ScanHistories",
                newName: "ScanHistory");

            migrationBuilder.AddPrimaryKey(
                name: "PK_ScanHistory",
                table: "ScanHistory",
                column: "Id");
        }
    }
}
