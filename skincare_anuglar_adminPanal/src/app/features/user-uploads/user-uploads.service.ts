import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, switchMap, tap } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class UserUploadsService {
  private baseUrl = 'https://localhost:7126/api';
  private token: string | null = null;

  constructor(private http: HttpClient) {}

  /**
   * Internal helper to guarantee a valid Admin bearer token before making API requests.
   * Performs automatic background login using the system admin credentials if not already authenticated.
   */
  private ensureAuthenticated(): Observable<string> {
    if (this.token) {
      return of(this.token);
    }
    return this.http.post<any>(`${this.baseUrl}/auth/login`, {
      email: 'admin@skinai.local',
      password: 'SecureAdminPassword123!'
    }).pipe(
      tap(res => {
        if (res && res.token) {
          this.token = res.token;
        }
      }),
      map(res => {
        if (!this.token) {
          throw new Error('Authentication failed: No token returned from server.');
        }
        return this.token;
      })
    );
  }

  /**
   * Fetches the summary of all scan histories from the backend dashboard endpoint.
   */
  getScanSummaryFeed(): Observable<any[]> {
    return this.ensureAuthenticated().pipe(
      switchMap(token => {
        const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);
        return this.http.get<any[]>(`${this.baseUrl}/AdminDashboard/scans/summary`, { headers });
      })
    );
  }

  /**
   * Fetches the detailed inspection diagnostic and heatmap payload for a specific scan ID.
   */
  getScanDetail(scanId: string): Observable<any> {
    return this.ensureAuthenticated().pipe(
      switchMap(token => {
        const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);
        return this.http.get<any>(`${this.baseUrl}/AdminDashboard/scans/${scanId}`, { headers });
      })
    );
  }
}
