import { Routes } from '@angular/router';
import { ClassifierFormComponent } from './classifier-form/classifier-form.component';

export const routes: Routes = [
  { path: 'classifier-form', component: ClassifierFormComponent },
  { path: '', redirectTo: '/classifier-form', pathMatch: 'full' },
];
