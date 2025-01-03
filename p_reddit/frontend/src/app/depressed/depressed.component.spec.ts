import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DepressedComponent } from './depressed.component';

describe('DepressedComponent', () => {
  let component: DepressedComponent;
  let fixture: ComponentFixture<DepressedComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DepressedComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DepressedComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
