import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ClassifierService {
  private apiUrl = 'http://localhost:5000/set_params';

  constructor(private http: HttpClient) { }

  setParams(params: any): Observable<any> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<any>(this.apiUrl, params, { headers });
  }
}