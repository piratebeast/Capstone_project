import { Component, ElementRef, OnInit, OnDestroy, ViewChild, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgIconComponent, provideIcons } from '@ng-icons/core';
import { heroEllipsisHorizontal, heroEye, heroMap, heroChartBar, heroSparkles, heroTrash } from '@ng-icons/heroicons/outline';
import { UserUploadsService } from './user-uploads.service';
import { ActivatedRoute } from '@angular/router';
import { Subscription, interval } from 'rxjs';
import { startWith, switchMap } from 'rxjs/operators';

@Component({
  selector: 'app-user-uploads',
  standalone: true,
  imports: [CommonModule, NgIconComponent],
  templateUrl: './user-uploads.component.html',
  viewProviders: [provideIcons({ heroEllipsisHorizontal, heroEye, heroMap, heroChartBar, heroSparkles, heroTrash })]
})
export class UserUploadsComponent implements OnInit, OnDestroy {
  @ViewChild('facialCanvas', { static: false }) facialCanvas!: ElementRef<HTMLCanvasElement>;

  // UI and Action trigger states
  isMenuOpen = false;
  showDeleteConfirmModal = false;
  isDeleting = false;
  // Critique & Routine states
  isCritiqueLoading = false;
  critiqueError: string | null = null;
  routineData: any = null;
  isRoutineLoading = false;
  routineError: string | null = null;
  activeRoutineTab: 'am' | 'pm' | 'weekly' = 'am';

  // Scans list and loading states
  scansSummaryList: any[] = [];
  private pollSubscription?: Subscription;
  selectedScanId: string | null = null;
  activeScanDetail: any = null;
  isLoading = false;
  isListLoading = false;
  globalConfidence = 0;

  patient = {
    name: 'Loading...',
    timestamp: '',
    source: 'Mobile App',
    imageUrl: '',
    avatarUrl: ''
  };

  metrics = {
    acneConfidence: 0,
    rednessConfidence: 0,
    darkSpotsConfidence: 0,
    wrinklesConfidence: 0,
    darkCirclesConfidence: 0
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
  private activeImage: HTMLImageElement | null = null;

  constructor(
    private userUploadsService: UserUploadsService,
    private route: ActivatedRoute
  ) { }

  ngOnInit() {
    this.route.queryParams.subscribe(params => {
      const queryScanId = params['scanId'];
      if (queryScanId) {
        this.selectedScanId = queryScanId;
        this.selectScan(queryScanId);
      }
    });
    this.loadScanSummaryFeed();
  }

  ngOnDestroy() {
    if (this.pollSubscription) {
      this.pollSubscription.unsubscribe();
    }
  }

  setRoutineTab(tab: 'am' | 'pm' | 'weekly') {
    this.activeRoutineTab = tab;
  }

  // Helper getters
  get isAnyLayerActive(): boolean {
    return Object.values(this.layers).some(layer => layer.active);
  }

  get isAllLayersActive(): boolean {
    return Object.values(this.layers).every(layer => layer.active);
  }

  /**
   * Load the scan summary feed list and set up real-time polling.
   */
  loadScanSummaryFeed() {
    this.isListLoading = this.scansSummaryList.length === 0;
    this.pollSubscription = interval(5000)
      .pipe(
        startWith(0),
        switchMap(() => this.userUploadsService.getScanSummaryFeed())
      )
      .subscribe({
        next: (data) => {
          const oldList = this.scansSummaryList;
          this.scansSummaryList = (data || []).map((newScan: any) => {
            const newId = newScan.scanId || newScan.ScanId;
            const existing = oldList.find(s => (s.scanId || s.ScanId) === newId);
            if (existing && (existing.aiCritique || existing.AiCritique)) {
              newScan.aiCritique = existing.aiCritique || existing.AiCritique;
              newScan.AiCritique = existing.aiCritique || existing.AiCritique;
            }
            return newScan;
          });
          this.isListLoading = false;
          if (this.scansSummaryList.length > 0 && !this.selectedScanId) {
            // Automatically select and load the first scan detail if none is selected
            this.selectScan(this.scansSummaryList[0].scanId || this.scansSummaryList[0].ScanId);
          }
        },
        error: (err) => {
          console.error('Failed to load scan summary feed:', err);
          this.isListLoading = false;
        }
      });
  }

  /**
   * Fetch details for a specific scan and calibrate properties.
   */
  selectScan(scanId: string) {
    if (!scanId) return;
    this.selectedScanId = scanId;
    this.isLoading = true;

    // Reset active heatmap layer states on loading new scan to avoid state leak
    this.showDetections = true;
    Object.keys(this.layers).forEach(key => {
      this.layers[key as keyof typeof this.layers].active = false;
    });

    this.isCritiqueLoading = false;
    this.critiqueError = null;

    this.routineData = null;
    this.isRoutineLoading = true;
    this.routineError = null;

    this.activeImage = null;
    this.clearCanvas();

    // Fetch Routine details
    this.userUploadsService.getScanRoutine(scanId).subscribe({
      next: (routine) => {
        this.routineData = routine;
        this.isRoutineLoading = false;
        this.routineError = null;
      },
      error: (err) => {
        console.error(`Failed to load routine for scan ID: ${scanId}`, err);
        this.isRoutineLoading = false;
        if (!this.routineData) {
          this.routineError = 'Routine information unavailable';
        }
      }
    });

    this.userUploadsService.getScanDetail(scanId).subscribe({
      next: (data) => {
        this.activeScanDetail = data;
        this.isLoading = false;

        const sched = data.regimenSchedule || data.RegimenSchedule;
        if (sched || data.dailyAm || data.DailyAm) {
          this.routineData = {
            routineClass: data.routineClass || data.RoutineClass,
            regimenSchedule: sched || {
              dailyAm: data.dailyAm || data.DailyAm,
              dailyPm: data.dailyPm || data.DailyPm,
              weeklyTreatments: data.weeklyTreatments || data.WeeklyTreatments
            }
          };
          this.isRoutineLoading = false;
          this.routineError = null;
        }

        // Ensure any existing AiCritique value from API or local list cache is parsed and retained
        const localScan = this.scansSummaryList.find(s => (s.scanId || s.ScanId) === scanId);
        const existingCritique = data.aiCritique || data.AiCritique || data.critiqueText || data.CritiqueText ||
          (localScan ? (localScan.aiCritique || localScan.AiCritique) : '') || '';

        if (existingCritique) {
          this.activeScanDetail.aiCritique = existingCritique;
          this.activeScanDetail.AiCritique = existingCritique;
          if (localScan) {
            localScan.aiCritique = existingCritique;
            localScan.AiCritique = existingCritique;
          }
        }

        // Step 1: Bind global confidence decimal (e.g. 0.9293 -> 92.93%)
        const rawConfidence = data.confidence !== undefined ? data.confidence : data.Confidence;
        this.globalConfidence = this.toPercent(rawConfidence);

        // Step 3: Read originalImageUrl and resolve absolute URL path
        const relativeUrl = data.originalImageUrl || data.OriginalImageUrl || '';
        const resolvedUrl = this.getFullImageUrl(relativeUrl);

        const patientName = this.getPatientName(data.userId || data.UserId);
        this.patient = {
          name: patientName,
          timestamp: this.formatScanDate(data.scanDate || data.ScanDate),
          source: 'Mobile App',
          imageUrl: resolvedUrl,
          avatarUrl: `https://ui-avatars.com/api/?name=${encodeURIComponent(patientName)}&background=E6F0FD&color=004B8F`
        };

        // Step 4: Calibrate individual diagnostics progress tracker metrics
        const diagnostics = data.diagnostics || data.Diagnostics || {};
        this.metrics = {
          acneConfidence: this.toPercent(diagnostics.acne ?? diagnostics.Acne ?? 0),
          darkSpotsConfidence: this.toPercent(diagnostics.darkSpots ?? diagnostics.DarkSpots ?? 0),
          wrinklesConfidence: this.toPercent(diagnostics.wrinkles ?? diagnostics.Wrinkles ?? 0),
          rednessConfidence: this.toPercent(diagnostics.redness ?? diagnostics.Redness ?? 0),
          darkCirclesConfidence: this.toPercent(diagnostics.darkCircles ?? diagnostics.DarkCircles ?? 0)
        };

        // Initialize rendering pipeline
        this.initAndDrawCanvas();
      },
      error: (err) => {
        console.error(`Failed to load scan details for ID: ${scanId}`, err);
        this.isLoading = false;
      }
    });
  }

  /**
   * Loads the HTMLImageElement from the patient image URL and draws it on canvas.
   */
  initAndDrawCanvas() {
    if (!this.patient.imageUrl) {
      this.activeImage = null;
      this.clearCanvas();
      return;
    }

    // Defer execution using setTimeout to let Angular render the canvas element in the DOM
    setTimeout(() => {
      if (!this.facialCanvas) {
        // Fallback retry if Angular change detection has not completed yet
        setTimeout(() => {
          if (this.facialCanvas) {
            this.loadAndDrawImageOnCanvas();
          } else {
            console.error('Facial canvas ViewChild remains undefined after retry.');
          }
        }, 100);
        return;
      }
      this.loadAndDrawImageOnCanvas();
    }, 50);
  }

  private loadAndDrawImageOnCanvas() {
    if (!this.facialCanvas) return;
    const canvas = this.facialCanvas.nativeElement;
    const img = new Image();
    img.onload = () => {
      this.activeImage = img;
      // Calibrate internal canvas resolution to match natural image size
      canvas.width = img.naturalWidth || 640;
      canvas.height = img.naturalHeight || 480;
      this.drawCanvasLayers();
    };
    img.onerror = () => {
      console.error('Failed to load image:', this.patient.imageUrl);
      this.activeImage = null;
      this.clearCanvas();
    };
    img.src = this.patient.imageUrl;
  }

  clearCanvas() {
    if (this.facialCanvas) {
      const canvas = this.facialCanvas.nativeElement;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = 640;
        canvas.height = 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#9CA3AF'; // Tailwind text-gray-400
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('No Image Available', canvas.width / 2, canvas.height / 2);
      }
    }
  }

  /**
   * Renders the base photo and overlays active heatmap layers.
   */
  drawCanvasLayers() {
    if (!this.facialCanvas || !this.activeImage) return;
    const canvas = this.facialCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear Canvas and Draw Layer 1: Face Photo
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(this.activeImage, 0, 0, canvas.width, canvas.height);

    // Step 5: Draw Layer 2 Canvas Pixel Blending for Active Heatmaps
    if (this.activeScanDetail) {
      const heatmaps = this.activeScanDetail.heatmaps || this.activeScanDetail.Heatmaps || {};
      const heatmapChannels = {
        acne: heatmaps.acne || heatmaps.Acne,
        darkSpots: heatmaps.darkSpots || heatmaps.DarkSpots,
        wrinkles: heatmaps.wrinkles || heatmaps.Wrinkles,
        redness: heatmaps.redness || heatmaps.Redness,
        darkCircles: heatmaps.darkCircles || heatmaps.DarkCircles
      };

      Object.entries(this.layers).forEach(([key, layer]) => {
        if (layer.active) {
          // Check if the target condition diagnostic score is greater than 0% before drawing the zone
          const confidenceKey = `${key}Confidence` as keyof typeof this.metrics;
          const score = this.metrics[confidenceKey];
          if (score > 0) {
            const flatArray = heatmapChannels[key as keyof typeof heatmapChannels];
            if (flatArray && flatArray.length === 50176) {
              this.blendHeatmapLayer(ctx, canvas.width, canvas.height, flatArray, layer.color);
            }
          }
        }
      });
    }
  }

  /**
   * Loops through flat heatmap array, filters ambient noise, creates a 224x224 offscreen image buffer,
   * and draws it back onto the primary canvas context using native linear interpolation scaling.
   */
  blendHeatmapLayer(ctx: CanvasRenderingContext2D, canvasWidth: number, canvasHeight: number, dataArray: number[], hexColor: string) {
    const offscreen = document.createElement('canvas');
    offscreen.width = 224;
    offscreen.height = 224;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    const rgb = this.hexToRgb(hexColor) || { r: 255, g: 0, b: 0 };
    const imgData = offCtx.createImageData(224, 224);
    const pixels = imgData.data;

    for (let i = 0; i < 50176; i++) {
      const val = dataArray[i];
      const offset = i * 4;

      if (val < 0.05) {
        // Step 5 Filter: Exclude low background ambient noise under 0.05
        pixels[offset] = 0;
        pixels[offset + 1] = 0;
        pixels[offset + 2] = 0;
        pixels[offset + 3] = 0;
      } else {
        pixels[offset] = rgb.r;
        pixels[offset + 1] = rgb.g;
        pixels[offset + 2] = rgb.b;
        // Translucent Alpha Blending mapping
        pixels[offset + 3] = Math.round(val * 255 * 0.7);
      }
    }

    offCtx.putImageData(imgData, 0, 0);
    // Scale 224x224 canvas directly on top of the main canvas
    ctx.drawImage(offscreen, 0, 0, canvasWidth, canvasHeight);
  }

  drawMockBoundingBoxes(ctx: CanvasRenderingContext2D, width: number, height: number) {
    // Draw visual indicators similar to original absolute divs
    ctx.save();
    // Acne detection box
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = Math.max(2, width * 0.003);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';
    const acneX = width * 0.35;
    const acneY = height * 0.40;
    const acneR = width * 0.08;
    ctx.beginPath();
    ctx.arc(acneX + acneR, acneY + acneR, acneR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Dark spots detection box
    ctx.strokeStyle = '#004B8F';
    ctx.fillStyle = 'rgba(0, 75, 143, 0.1)';
    const dsX = width * 0.55;
    const dsY = height * 0.65;
    const dsR = width * 0.06;
    ctx.beginPath();
    ctx.arc(dsX + dsR, dsY + dsR, dsR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }

  toggleLayer(layerKey: keyof typeof this.layers) {
    this.showDetections = false;
    this.layers[layerKey].active = !this.layers[layerKey].active;
    this.drawCanvasLayers();
  }

  showOriginal() {
    this.showDetections = false;
    Object.keys(this.layers).forEach(key => {
      this.layers[key as keyof typeof this.layers].active = false;
    });
    this.drawCanvasLayers();
  }

  toggleAllHeatmaps() {
    this.showDetections = false;
    const targetState = !this.isAllLayersActive;
    Object.keys(this.layers).forEach(key => {
      this.layers[key as keyof typeof this.layers].active = targetState;
    });
    this.drawCanvasLayers();
  }

  resetToDefault() {
    this.showDetections = true;
    Object.keys(this.layers).forEach(key => {
      this.layers[key as keyof typeof this.layers].active = false;
    });
    this.drawCanvasLayers();
  }

  // --- Helper parsing utilities ---

  private toPercent(val: any): number {
    const num = parseFloat(val);
    if (isNaN(num)) return 0;
    if (num >= 0 && num <= 1.0) {
      return Math.round(num * 10000) / 100; // Drive standard decimal scalar
    }
    return Math.round(num);
  }

  private getFullImageUrl(url: string): string {
    if (!url) return '';
    if (url.startsWith('http')) return url;
    return `https://localhost:7126${url.startsWith('/') ? '' : '/'}${url}`;
  }

  private getPatientName(userId: string): string {
    if (!userId) return 'Sarah J.';
    const names = ['Sarah J.', 'Michael K.', 'David L.', 'Emma W.', 'Sophia M.', 'James B.'];
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      hash = userId.charCodeAt(i) + ((hash << 5) - hash);
    }
    const index = Math.abs(hash) % names.length;
    return names[index];
  }

  formatScanDate(dateStr: string): string {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return dateStr;
    }
  }

  private hexToRgb(hex: string): { r: number, g: number, b: number } | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent) {
    this.isMenuOpen = false;
  }

  toggleMenu(event: Event) {
    event.stopPropagation();
    this.isMenuOpen = !this.isMenuOpen;
  }

  triggerDelete(event: Event) {
    event.stopPropagation();
    this.isMenuOpen = false;
    this.showDeleteConfirmModal = true;
  }

  cancelDelete() {
    this.showDeleteConfirmModal = false;
  }

  confirmDelete() {
    const deletedId = this.selectedScanId;
    if (!deletedId) return;

    this.isDeleting = true;
    this.userUploadsService.deleteScan(deletedId).subscribe({
      next: () => {
        // Find index of the deleted item in the summary list
        const deletedIdx = this.scansSummaryList.findIndex(s => (s.scanId || s.ScanId) === deletedId);

        // Remove from list
        this.scansSummaryList = this.scansSummaryList.filter(s => (s.scanId || s.ScanId) !== deletedId);

        this.showDeleteConfirmModal = false;
        this.isDeleting = false;

        if (this.scansSummaryList.length > 0) {
          // Find next available item slot
          let nextIdx = deletedIdx;
          if (nextIdx >= this.scansSummaryList.length) {
            nextIdx = this.scansSummaryList.length - 1;
          }
          const nextScan = this.scansSummaryList[nextIdx];
          const nextScanId = nextScan.scanId || nextScan.ScanId;
          this.selectScan(nextScanId);
        } else {
          this.selectedScanId = null;
          this.activeScanDetail = null;
        }
      },
      error: (err) => {
        console.error('Failed to delete scan:', err);
        this.isDeleting = false;
        alert('Failed to delete the scan record. Please try again.');
      }
    });
  }

  get critiqueInfo(): { hasCritique: boolean; succeeded: boolean; critiqueText: string; errorMessage: string } {
    return this.getCritiqueInfo(this.activeScanDetail);
  }

  getCritiqueInfo(scanDetail: any): { hasCritique: boolean; succeeded: boolean; critiqueText: string; errorMessage: string } {
    if (!scanDetail) {
      return { hasCritique: false, succeeded: false, critiqueText: '', errorMessage: '' };
    }

    const rawCritique = scanDetail.aiCritique ?? scanDetail.AiCritique;
    const directText = scanDetail.critiqueText ?? scanDetail.CritiqueText;
    const directError = scanDetail.errorMessage ?? scanDetail.ErrorMessage;

    let explicitSucceeded: boolean | undefined = undefined;
    if (typeof rawCritique === 'object' && rawCritique !== null) {
      if (rawCritique.succeeded !== undefined) explicitSucceeded = Boolean(rawCritique.succeeded);
      else if (rawCritique.Succeeded !== undefined) explicitSucceeded = Boolean(rawCritique.Succeeded);
    }
    if (explicitSucceeded === undefined) {
      if (scanDetail.succeeded !== undefined) explicitSucceeded = Boolean(scanDetail.succeeded);
      else if (scanDetail.Succeeded !== undefined) explicitSucceeded = Boolean(scanDetail.Succeeded);
      else if (scanDetail.critiqueSucceeded !== undefined) explicitSucceeded = Boolean(scanDetail.critiqueSucceeded);
      else if (scanDetail.CritiqueSucceeded !== undefined) explicitSucceeded = Boolean(scanDetail.CritiqueSucceeded);
    }

    if (typeof rawCritique === 'object' && rawCritique !== null) {
      const succeeded = explicitSucceeded ?? true;
      const text = rawCritique.critiqueText ?? rawCritique.CritiqueText ?? directText ?? '';
      const err = rawCritique.errorMessage ?? rawCritique.ErrorMessage ?? directError ?? '';
      return {
        hasCritique: true,
        succeeded,
        critiqueText: text,
        errorMessage: err || text || 'AI Model Critique temporarily unavailable.'
      };
    }

    const textStr = (typeof rawCritique === 'string' ? rawCritique : (directText || '')).trim();
    const errorStr = (directError || '').trim();

    if (!textStr && !errorStr && explicitSucceeded === undefined) {
      return { hasCritique: false, succeeded: false, critiqueText: '', errorMessage: '' };
    }

    let succeeded = explicitSucceeded;
    if (succeeded === undefined) {
      const lower = textStr.toLowerCase();
      if (lower.includes('temporarily unavailable') || lower.includes('failed') || lower.includes('error') || errorStr.length > 0) {
        succeeded = false;
      } else {
        succeeded = true;
      }
    }

    return {
      hasCritique: true,
      succeeded,
      critiqueText: textStr,
      errorMessage: errorStr || textStr || 'AI Model Critique temporarily unavailable.'
    };
  }

  /**
   * Dispatches the API request to run AI clinical audit critique feedback.
   */
  runAiCritique() {
    const scanId = this.selectedScanId;
    if (!scanId) return;

    this.isCritiqueLoading = true;
    this.critiqueError = null;

    this.userUploadsService.triggerScanCritique(scanId).subscribe({
      next: (res) => {
        this.isCritiqueLoading = false;

        const succeeded = res.succeeded !== undefined ? res.succeeded : res.Succeeded;
        const critiqueText = res.critiqueText !== undefined ? res.critiqueText : res.CritiqueText;
        const errorMessage = res.errorMessage !== undefined ? res.errorMessage : res.ErrorMessage;

        const isSuccess = succeeded ?? (errorMessage ? false : true);
        const critiqueObj = {
          succeeded: isSuccess,
          critiqueText: critiqueText || '',
          errorMessage: errorMessage || (isSuccess ? '' : critiqueText || 'An error occurred during critique generation.')
        };

        if (this.activeScanDetail) {
          this.activeScanDetail.aiCritique = critiqueObj;
          this.activeScanDetail.AiCritique = critiqueObj;
          this.activeScanDetail.succeeded = isSuccess;
          this.activeScanDetail.critiqueText = critiqueObj.critiqueText;
          this.activeScanDetail.errorMessage = critiqueObj.errorMessage;
        }
        const summaryItem = this.scansSummaryList.find(s => (s.scanId || s.ScanId) === scanId);
        if (summaryItem) {
          summaryItem.aiCritique = critiqueObj;
          summaryItem.AiCritique = critiqueObj;
        }

        if (!isSuccess) {
          this.critiqueError = critiqueObj.errorMessage;
        }
      },
      error: (err) => {
        this.isCritiqueLoading = false;
        const errMsg = err.error?.errorMessage || err.error?.ErrorMessage || err.error?.critiqueText || err.message || 'Failed to trigger critique generation.';
        this.critiqueError = errMsg;

        const critiqueObj = {
          succeeded: false,
          critiqueText: '',
          errorMessage: errMsg
        };

        if (this.activeScanDetail) {
          this.activeScanDetail.aiCritique = critiqueObj;
          this.activeScanDetail.AiCritique = critiqueObj;
          this.activeScanDetail.succeeded = false;
          this.activeScanDetail.errorMessage = errMsg;
        }
      }
    });
  }

  /**
   * Parses the clinical critique block of text into structured HTML with red/yellow/green indicators and bolded terms.
   */
  getStructuredCritiqueHtml(text: string): string {
    if (!text) return '';

    // Split sentences
    const sentences = text.split(/(?<=\.|\?|!)\s+/);

    let html = '<div class="space-y-3.5">';

    for (let sentence of sentences) {
      sentence = sentence.trim();
      if (!sentence) continue;

      // Determine prefix icon & text color class based on context/sentiment analysis
      let iconColor = 'bg-gray-400';
      let textClass = 'text-gray-700';

      const lower = sentence.toLowerCase();
      if (lower.includes('correctly') || lower.includes('aligning') || lower.includes('successful')) {
        iconColor = 'bg-[#10B981]'; // Emerald/Green for correct inferences
      } else if (lower.includes('over-diagnoses') || lower.includes('misinterpreting') || lower.includes('likely') || lower.includes('however')) {
        iconColor = 'bg-[#F59E0B]'; // Amber/Yellow for overdiagnosis/misinterpretation warnings
      } else if (lower.includes('inaccurate') || lower.includes('lacks') || lower.includes('failed') || lower.includes('consequently')) {
        iconColor = 'bg-[#EF4444]'; // Red for critical recommendation failures / errors
        textClass = 'text-gray-900 font-semibold';
      }

      // Highlight key technical terms
      const termsToHighlight = [
        'high hyperpigmentation', 'hyperpigmentation', 'facial ephelides',
        'acne, erythema, and dark circles', 'acne', 'erythema', 'dark circles',
        'concentrated freckle pattern', 'freckle pattern',
        'inflammatory lesions', 'vascular redness',
        'acne and redness control routine', 'inflammatory pathology'
      ];

      let highlighted = sentence;
      termsToHighlight.forEach(term => {
        const escapedTerm = term.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
        const reg = new RegExp(`\\b${escapedTerm}\\b`, 'gi');
        highlighted = highlighted.replace(reg, (match) => `<strong class="text-gray-950 font-bold">${match}</strong>`);
      });

      html += `
        <div class="flex gap-2.5 items-start">
          <span class="w-1.5 h-1.5 rounded-full ${iconColor} mt-1.5 shrink-0 shadow-sm"></span>
          <p class="text-xs ${textClass} leading-relaxed">${highlighted}</p>
        </div>
      `;
    }

    html += '</div>';
    return html;
  }
}
