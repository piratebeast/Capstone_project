import { Routes } from '@angular/router';
import { MainLayoutComponent } from './layout/main-layout/main-layout.component';
import { authGuard } from './core/guards/auth.guard';

export const routes: Routes = [
  {
    path: 'login',
    loadComponent: () => import('./features/auth/login/login.component').then(c => c.LoginComponent)
  },
  {
    path: '',
    component: MainLayoutComponent,
    canActivate: [authGuard],
    children: [
      { path: '', redirectTo: 'overview', pathMatch: 'full' },
      { path: 'overview', loadComponent: () => import('./features/overview/overview.component').then(c => c.OverviewComponent) },
      { path: 'performance', loadComponent: () => import('./features/performance/performance.component').then(c => c.PerformanceComponent) },
      { path: 'user-uploads', loadComponent: () => import('./features/user-uploads/user-uploads.component').then(c => c.UserUploadsComponent) },
    ]
  },
  { path: '**', redirectTo: 'login' }
];
