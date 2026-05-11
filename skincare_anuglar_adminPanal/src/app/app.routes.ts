import { Routes } from '@angular/router';
import { MainLayoutComponent } from './layout/main-layout/main-layout.component';

export const routes: Routes = [
  {
    path: '',
    component: MainLayoutComponent,
    children: [
      { path: '', redirectTo: 'overview', pathMatch: 'full' },
      { path: 'overview', loadComponent: () => import('./features/overview/overview.component').then(c => c.OverviewComponent) },
      { path: 'performance', loadComponent: () => import('./features/performance/performance.component').then(c => c.PerformanceComponent) },
      { path: 'user-uploads', loadComponent: () => import('./features/user-uploads/user-uploads.component').then(c => c.UserUploadsComponent) },
    ]
  }
];
