import { Component } from '@angular/core';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroBeaker, heroArrowRightOnRectangle } from '@ng-icons/heroicons/outline';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-top-nav',
  standalone: true,
  imports: [NgIconComponent],
  templateUrl: './top-nav.component.html',
  viewProviders: [provideIcons({ heroBeaker, heroArrowRightOnRectangle })]
})
export class TopNavComponent {
  currentUser: any = null;

  constructor(private authService: AuthService) {
    this.currentUser = this.authService.getCurrentUser();
  }

  logout() {
    this.authService.logout();
  }
}
