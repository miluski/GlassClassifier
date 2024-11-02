import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ClassifierService } from '../classifier.service';

@Component({
  selector: 'app-classifier-form',
  templateUrl: './classifier-form.component.html',
  standalone: true,
  imports: [CommonModule, FormsModule],
  styleUrls: ['./classifier-form.component.css'],
})
export class ClassifierFormComponent {
  params: any = {
    hidden_layer_sizes: [100],
    learning_rate_init: 0.1,
    max_iter: 2000,
    tol: 0.0001,
    activation: 'logistic',
    solver: 'adam',
    alpha: 0.01,
    batch_size: 'auto',
  };
  responseMessage: string = '';
  loading: boolean = false;
  keyLabels: { [key: string]: string } = {
    activation: 'Funkcja aktywacji',
    alpha: 'Współczynnik alpha',
    batch_size: 'Wielkość podzbiorów',
    solver: 'Solver',
    hidden_layer_sizes: 'Rozmiary warstw ukrytych',
    learning_rate_init: 'Początkowy współczynnik uczenia',
    max_iter: 'Liczba epok',
    tol: 'Współczynnik tolerancji',
  };
  activationOptions: string[] = ['relu', 'tanh', 'logistic', 'identity'];
  solverOptions: string[] = ['adam', 'sgd'];
  batchSizeOptions: string[] = ['auto', '32', '64', '128'];

  constructor(
    private classifierService: ClassifierService,
    private router: Router
  ) {}

  convertArrayToString(array: number[]): string {
    if (array.length === 1) {
      return `(${array[0]},)`;
    }
    return `(${array.join(',')})`;
  }
  setParams() {
    this.loading = true;
    const paramsToSend = {
      ...this.params,
      hidden_layer_sizes: this.convertArrayToString(
        this.params.hidden_layer_sizes
      ),
    };
    if (paramsToSend.batch_size !== 'auto') {
      paramsToSend.batch_size = parseInt(paramsToSend.batch_size as string, 10);
    }
    this.classifierService.setParams(paramsToSend).subscribe(
      (response) => {
        this.loading = false;
        this.classifierService.parameters = response;
        this.router.navigate(['/results']);
      },
      (error) => {
        this.responseMessage = `Error: ${error.error.error}`;
        this.loading = false;
      }
    );
  }
  getLabel(key: string): string {
    return this.keyLabels[key] || key;
  }
  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }
}
