import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ClassifierService {
  private apiUrl = 'http://192.168.0.11:5005/set_params';
  parameters: any;

  constructor(private http: HttpClient) {}

  setParams(params: any): Observable<any> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<any>(this.apiUrl, params, { headers });
  }
}
