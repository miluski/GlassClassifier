import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { catchError, of, tap } from 'rxjs';
import { ClassifierService } from '../classifier.service';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-classifier-form',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './classifier-form.component.html',
  styleUrl: './classifier-form.component.css',
})
export class ClassifierFormComponent {
  objectKeys = Object.keys;

  params: {
    [key: string]: any;
    activation: string;
    alpha: number;
    batch_size: string;
    hidden_layer_sizes: number[];
    learning_rate_init: number;
    max_iter: number;
    solver: string;
    tol: number;
  } = {
    activation: 'relu',
    alpha: 0.001,
    batch_size: 'auto',
    hidden_layer_sizes: [100],
    learning_rate_init: 0.01,
    max_iter: 2000,
    solver: 'adam',
    tol: 0.0001,
  };

  responseMessage: string = '';

  constructor(private classifierService: ClassifierService) {}

  setParams(): void {
    this.classifierService
      .setParams(this.params)
      .pipe(
        tap((response) => {
          this.responseMessage = response.message;
        }),
        catchError((error) => {
          this.responseMessage = error.error
            ? error.error.error
            : 'An error occurred';
          return of(null);
        })
      )
      .subscribe();
  }
}
