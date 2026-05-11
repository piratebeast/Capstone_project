import { Component } from '@angular/core';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroEllipsisHorizontal, heroEye, heroMap, heroChartBar, heroSparkles } from '@ng-icons/heroicons/outline';

@Component({
  selector: 'app-user-uploads',
  standalone: true,
  imports: [NgIconComponent],
  templateUrl: './user-uploads.component.html',
  viewProviders: [provideIcons({ heroEllipsisHorizontal, heroEye, heroMap, heroChartBar, heroSparkles })]
})
export class UserUploadsComponent {
  patient = {
    name: 'Sarah J.',
    timestamp: 'Today, 10:42 AM',
    source: 'Mobile App',
    imageUrl: 'https://images.unsplash.com/photo-1512290923902-8a9f81dc236c?auto=format&fit=crop&q=80&w=800'
  };

  metrics = {
    acneConfidence: 75,
    rednessConfidence: 20,
    darkSpotsConfidence: 15
  };
}
