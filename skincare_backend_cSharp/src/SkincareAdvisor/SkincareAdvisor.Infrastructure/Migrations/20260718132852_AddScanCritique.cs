using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SkincareAdvisor.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddScanCritique : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ScanCritiques",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    ScanHistoryId = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    ModelUsed = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    CritiqueText = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    RawResponseJson = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    GeneratedAt = table.Column<DateTime>(type: "datetime2", nullable: false),
                    Succeeded = table.Column<bool>(type: "bit", nullable: false),
                    ErrorMessage = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ScanCritiques", x => x.Id);
                    table.ForeignKey(
                        name: "FK_ScanCritiques_ScanHistories_ScanHistoryId",
                        column: x => x.ScanHistoryId,
                        principalTable: "ScanHistories",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_ScanCritiques_ScanHistoryId",
                table: "ScanCritiques",
                column: "ScanHistoryId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "ScanCritiques");
        }
    }
}
