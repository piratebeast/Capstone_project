import { Component } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions } from 'chart.js';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroArrowTrendingUp, heroDocumentCheck, heroAdjustmentsHorizontal, heroCog } from '@ng-icons/heroicons/outline';

@Component({
  selector: 'app-performance',
  standalone: true,
  imports: [BaseChartDirective, NgIconComponent],
  templateUrl: './performance.component.html',
  viewProviders: [provideIcons({ heroArrowTrendingUp, heroDocumentCheck, heroAdjustmentsHorizontal, heroCog })]
})
export class PerformanceComponent {
  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: ['1', '5', '10', '15', '20', '25', '30'],
    datasets: [
      {
        data: [0.03, 0.04, 0.06, 0.05, 0.07, 0.09, 0.11],
        label: 'Feature Drift (Stage 2)',
        borderColor: '#004B8F',
        backgroundColor: 'transparent',
        tension: 0.4,
        borderWidth: 3,
        pointRadius: 0
      },
      {
        data: [0.01, 0.02, 0.01, 0.02, 0.03, 0.06, 0.16],
        label: 'Concept Drift (Stage 3)',
        borderColor: '#059669',
        backgroundColor: 'transparent',
        borderDash: [8, 4],
        tension: 0.4,
        borderWidth: 3,
        pointRadius: 0
      }
    ]
  };

  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        align: 'center',
        labels: {
          usePointStyle: false,
          boxWidth: 20,
          padding: 15,
          font: { family: 'Inter, sans-serif', size: 12 }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 0.25,
        grid: {
          color: '#E5E7EB'
        },
        ticks: {
          stepSize: 0.1,
          color: '#6B7280'
        }
      },
      x: {
        grid: {
          color: '#E5E7EB'
        },
        ticks: {
          display: false
        }
      }
    }
  };
}
