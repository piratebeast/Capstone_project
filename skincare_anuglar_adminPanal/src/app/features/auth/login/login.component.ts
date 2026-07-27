import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../../core/services/auth.service';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroBeaker, heroLockClosed, heroEnvelope, heroEye, heroEyeSlash, heroShieldCheck, heroSparkles, heroExclamationTriangle } from '@ng-icons/heroicons/outline';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, NgIconComponent],
  templateUrl: './login.component.html',
  viewProviders: [provideIcons({ heroBeaker, heroLockClosed, heroEnvelope, heroEye, heroEyeSlash, heroShieldCheck, heroSparkles, heroExclamationTriangle })]
})
export class LoginComponent implements OnInit {
  email = 'admin@skinai.local';
  password = 'SecureAdminPassword123!';
  showPassword = false;
  isLoading = false;
  errorMessage: string | null = null;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit() {
    // If already logged in, redirect straight to dashboard overview
    if (this.authService.hasToken()) {
      this.router.navigate(['/overview']);
    }
  }

  togglePasswordVisibility() {
    this.showPassword = !this.showPassword;
  }

  onSubmit() {
    if (!this.email || !this.password) {
      this.errorMessage = 'Please enter both email and password.';
      return;
    }

    this.isLoading = true;
    this.errorMessage = null;

    this.authService.login(this.email, this.password).subscribe({
      next: () => {
        this.isLoading = false;
        this.router.navigate(['/overview']);
      },
      error: (err) => {
        this.isLoading = false;
        console.error('Login failed:', err);
        this.errorMessage = err.error?.message || err.error?.ErrorMessage || err.message || 'Invalid admin credentials or server authentication error.';
      }
    });
  }

  fillAdminCredentials() {
    this.email = 'admin@skinai.local';
    this.password = 'SecureAdminPassword123!';
    this.errorMessage = null;
  }
}
