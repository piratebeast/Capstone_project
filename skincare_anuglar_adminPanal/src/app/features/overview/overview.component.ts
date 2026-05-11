import { Component } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions } from 'chart.js';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroArrowTrendingUp, heroFunnel, heroClock, heroUsers, heroCheckCircle, heroExclamationTriangle, heroArrowRight } from '@ng-icons/heroicons/outline';

@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [BaseChartDirective, NgIconComponent],
  templateUrl: './overview.component.html',
  viewProviders: [provideIcons({ heroArrowTrendingUp, heroFunnel, heroClock, heroUsers, heroCheckCircle, heroExclamationTriangle, heroArrowRight })]
})
export class OverviewComponent {
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
}
