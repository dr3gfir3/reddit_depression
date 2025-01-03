import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EvaluateTextComponent } from './evaluate-text.component';

describe('EvaluateTextComponent', () => {
  let component: EvaluateTextComponent;
  let fixture: ComponentFixture<EvaluateTextComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EvaluateTextComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EvaluateTextComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
