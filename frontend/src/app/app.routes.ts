import { Routes } from '@angular/router';
import { ClassifierFormComponent } from './classifier-form/classifier-form.component';
import { ResultsComponent } from './results/results.component';

export const routes: Routes = [
  { path: 'classifier-form', component: ClassifierFormComponent },
  { path: '', redirectTo: '/classifier-form', pathMatch: 'full' },
  { path: 'results', component: ResultsComponent },
];
