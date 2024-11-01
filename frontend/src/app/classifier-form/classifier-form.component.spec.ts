import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClassifierFormComponent } from './classifier-form.component';

describe('ClassifierFormComponent', () => {
  let component: ClassifierFormComponent;
  let fixture: ComponentFixture<ClassifierFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ClassifierFormComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ClassifierFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
