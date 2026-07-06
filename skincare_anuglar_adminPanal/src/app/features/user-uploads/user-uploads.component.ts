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
    darkSpotsConfidence: 15,
    wrinklesConfidence: 42,
    darkCirclesConfidence: 58
  };

  // Heatmap layer states
  layers = {
    acne: { name: 'Acne', color: '#EF4444', active: false },
    darkSpots: { name: 'Dark Spots', color: '#F59E0B', active: false },
    wrinkles: { name: 'Wrinkles', color: '#10B981', active: false },
    redness: { name: 'Redness', color: '#EC4899', active: false },
    darkCircles: { name: 'Dark Circles', color: '#6366F1', active: false }
  };

  showDetections = true;

  // Helper getters
  get isAnyLayerActive(): boolean {
    return Object.values(this.layers).some(layer => layer.active);
  }

  get isAllLayersActive(): boolean {
    return Object.values(this.layers).every(layer => layer.active);
  }

  toggleLayer(layerKey: keyof typeof this.layers) {
    this.showDetections = false;
    this.layers[layerKey].active = !this.layers[layerKey].active;
  }

  showOriginal() {
    this.showDetections = false;
    Object.keys(this.layers).forEach(key => {
      this.layers[key as keyof typeof this.layers].active = false;
    });
  }

  toggleAllHeatmaps() {
    this.showDetections = false;
    const targetState = !this.isAllLayersActive;
    Object.keys(this.layers).forEach(key => {
      this.layers[key as keyof typeof this.layers].active = targetState;
    });
  }

  resetToDefault() {
    this.showDetections = true;
    Object.keys(this.layers).forEach(key => {
      this.layers[key as keyof typeof this.layers].active = false;
    });
  }
}
