import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ClassifierService } from '../classifier.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-results',
  templateUrl: './results.component.html',
  standalone: true,
  imports: [CommonModule],
  styleUrls: ['./results.component.css'],
})
export class ResultsComponent implements OnInit {
  parameters: any;

  constructor(
    public classifierService: ClassifierService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.parameters = this.classifierService.parameters;
    if (!this.parameters) {
      this.router.navigate(['/']);
    }
  }

  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }

  convertKeyToReadableFormat(key: string): string {
    const mapping: { [key: string]: string } = {
      first_validation: 'Pierwsza walidacja krzyżowa',
      second_validation: 'Druga walidacja krzyżowa',
      third_validation: 'Trzecia walidacja krzyżowa',
      fourth_validation: 'Czwarta walidacja krzyżowa',
      fifth_validation: 'Piąta walidacja krzyżowa',
    };
    return mapping[key as keyof typeof mapping] || key;
  }
}