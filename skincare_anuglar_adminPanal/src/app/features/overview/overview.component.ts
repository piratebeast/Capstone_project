import { Component, OnInit } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions } from 'chart.js';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroArrowTrendingUp, heroFunnel, heroClock, heroUsers, heroCheckCircle, heroExclamationTriangle, heroArrowRight } from '@ng-icons/heroicons/outline';
import { CommonModule } from '@angular/common';
import { UserUploadsService } from '../user-uploads/user-uploads.service';
import { RouterLink, Router } from '@angular/router';

@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [CommonModule, BaseChartDirective, NgIconComponent, RouterLink],
  templateUrl: './overview.component.html',
  viewProviders: [provideIcons({ heroArrowTrendingUp, heroFunnel, heroClock, heroUsers, heroCheckCircle, heroExclamationTriangle, heroArrowRight })]
})
export class OverviewComponent implements OnInit {
  public barChartData: ChartConfiguration<'bar'>['data'] = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    datasets: [
      { 
        data: [35, 45, 95, 60, 110, 85], 
        label: '.NET Coordinator',
        backgroundColor: '#004B8F',
        borderRadius: 2,
        barPercentage: 0.8,
        categoryPercentage: 0.8
      },
      { 
        data: [30, 42, 85, 55, 95, 75], 
        label: 'Python AI Engine',
        backgroundColor: '#D1E3F8',
        borderRadius: 2,
        barPercentage: 0.8,
        categoryPercentage: 0.8
      }
    ]
  };

  public barChartOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        align: 'start',
        labels: {
          usePointStyle: true,
          boxWidth: 8,
          padding: 20,
          font: { family: 'Inter, sans-serif', size: 12 }
        }
      }
    },
    scales: {
      y: {
        display: false,
        beginAtZero: true,
      },
      x: {
        grid: {
          display: false
        },
        ticks: {
          font: { family: 'Inter, sans-serif', size: 12 },
          color: '#6B7280'
        },
        border: { display: false }
      }
    }
  };

  metricsData: any = null;
  recentScans: any[] = [];

  constructor(
    private userUploadsService: UserUploadsService,
    private router: Router
  ) {}

  ngOnInit() {
    this.userUploadsService.getSystemOverviewMetrics().subscribe({
      next: (data) => {
        this.metricsData = data;
        
        // Update bar chart data dynamically if throughputData is present
        if (data && data.throughputData) {
          const throughput = data.throughputData;
          this.barChartData = {
            ...this.barChartData,
            datasets: [
              { ...this.barChartData.datasets[0], data: throughput },
              { ...this.barChartData.datasets[1], data: throughput.map((v: number) => Math.round(v * 0.85)) }
            ]
          };
        }
      },
      error: (err) => {
        console.error('Failed to load system telemetry metrics:', err);
      }
    });

    this.userUploadsService.getScanSummaryFeed().subscribe({
      next: (scans) => {
        this.recentScans = (scans || []).slice(0, 3);
      },
      error: (err) => {
        console.error('Failed to load recent activity feed:', err);
      }
    });
  }

  navigateToScan(scanId: string) {
    this.router.navigate(['/user-uploads'], { queryParams: { scanId } });
  }

  formatScanDate(dateStr: string): string {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return dateStr;
    }
  }
}
