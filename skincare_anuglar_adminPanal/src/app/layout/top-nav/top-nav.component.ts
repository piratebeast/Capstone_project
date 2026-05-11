import { Component } from '@angular/core';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroMagnifyingGlass, heroBell, heroCog8Tooth, heroQuestionMarkCircle, heroBeaker } from '@ng-icons/heroicons/outline';

@Component({
  selector: 'app-top-nav',
  standalone: true,
  imports: [NgIconComponent],
  templateUrl: './top-nav.component.html',
  viewProviders: [provideIcons({ heroMagnifyingGlass, heroBell, heroCog8Tooth, heroQuestionMarkCircle, heroBeaker })]
})
export class TopNavComponent {}
