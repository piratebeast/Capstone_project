import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroSquares2x2, heroUsers, heroChartBar, heroBeaker } from '@ng-icons/heroicons/outline';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [RouterModule, NgIconComponent],
  templateUrl: './sidebar.component.html',
  viewProviders: [provideIcons({ heroSquares2x2, heroUsers, heroChartBar, heroBeaker })]
})
export class SidebarComponent {
  navItems = [
    { label: 'Overview', route: '/overview', icon: 'heroSquares2x2' },
    { label: 'User Uploads', route: '/user-uploads', icon: 'heroUsers' },
    { label: 'Model Performance', route: '/performance', icon: 'heroChartBar' }
  ];
}
